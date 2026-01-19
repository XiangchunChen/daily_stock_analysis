# -*- coding: utf-8 -*-
"""
===================================
YfinanceFetcher - 兜底数据源 (Priority 4)
===================================

数据来源：Yahoo Finance（通过 yfinance 库）
特点：国际数据源、可能有延迟或缺失
定位：当所有国内数据源都失败时的最后保障

关键策略：
1. 自动将 A 股代码转换为 yfinance 格式（.SS / .SZ）
2. 处理 Yahoo Finance 的数据格式差异
3. 失败后指数退避重试
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from .base import BaseFetcher, DataFetchError, RateLimitError, STANDARD_COLUMNS, RealtimeQuote

logger = logging.getLogger(__name__)


class YfinanceFetcher(BaseFetcher):
    """
    Yfinance 数据源 (Priority 4)
    
    特点：
    - 支持全球股市（美股、港股、A股）
    - 数据质量较高
    - 访问速度较慢（需网络支持）
    
    作为兜底数据源，主要用于获取其他源不支持的股票数据
    """
    
    name: str = "YfinanceFetcher"
    priority: int = 4
    
    def __init__(self):
        """初始化 YfinanceFetcher"""
        pass
    
    def _convert_stock_code(self, stock_code: str) -> str:
        """
        转换股票代码为 Yahoo Finance 格式
        
        Yahoo Finance A 股代码格式：
        - 沪市：600519.SS (Shanghai Stock Exchange)
        - 深市：000001.SZ (Shenzhen Stock Exchange)
        - 美股：AAPL, MSFT (直接使用代码)
        
        Args:
            stock_code: 原始代码，如 '600519', '000001', 'AAPL'
            
        Returns:
            Yahoo Finance 格式代码
        """
        code = stock_code.strip()
        
        # 美股代码（纯字母）直接返回
        # 排除包含 .SS .SZ 的情况 (虽然 alpha check 也会排除点号)
        # 注意：有些美股代码包含点号，如 BRK.B
        is_us_stock = (
            code.replace('.', '').isalpha() and 
            not code.endswith(('.SS', '.SZ', '.SH'))
        )
        if is_us_stock:
            return code.upper()
        
        # 已经包含后缀的情况
        if '.SS' in code.upper() or '.SZ' in code.upper():
            return code.upper()
        
        # 去除可能的后缀
        code = code.replace('.SH', '').replace('.sh', '')
        
        # 根据代码前缀判断市场
        if code.startswith(('600', '601', '603', '688')):
            return f"{code}.SS"
        elif code.startswith(('000', '002', '300')):
            return f"{code}.SZ"
        else:
            # 如果是纯字母但没匹配上前面的逻辑，可能是生僻美股或其它，尝试直接用
            if code.replace('.', '').isalpha():
                return code.upper()
                
            logger.warning(f"无法确定股票 {code} 的市场，默认使用深市")
            return f"{code}.SZ"
            
    def get_realtime_quote(self, stock_code: str) -> Optional[RealtimeQuote]:
        """获取实时行情"""
        try:
             import yfinance as yf
             # Convert logic for symbol
             symbol = self._convert_stock_code(stock_code)
             ticker = yf.Ticker(symbol)
             
             # fast_info is preferred for price
             fast_info = ticker.fast_info
             price = float(fast_info.last_price) if fast_info.last_price else 0.0
             prev_close = float(fast_info.previous_close) if fast_info.previous_close else 0.0
             
             if price and prev_close:
                 change_amount = price - prev_close
                 change_pct = (change_amount / prev_close) * 100
             else:
                 change_amount = 0.0
                 change_pct = 0.0
             
             # basic info
             name = stock_code
             pe = 0.0
             pb = 0.0
             total_mv = 0.0
             high_52w = 0.0
             low_52w = 0.0
             
             try:
                 # info request might be slow
                 try:
                     info = ticker.info
                 except:
                     info = {}
                     
                 name = info.get('shortName') or info.get('longName') or stock_code
                 pe = info.get('trailingPE', 0.0) or 0.0
                 pb = info.get('priceToBook', 0.0) or 0.0
                 total_mv = info.get('marketCap', 0.0) or 0.0
                 high_52w = info.get('fiftyTwoWeekHigh', 0.0) or 0.0
                 low_52w = info.get('fiftyTwoWeekLow', 0.0) or 0.0
             except Exception:
                 pass
                 
             # Fallback to fast_info
             if total_mv == 0 and hasattr(fast_info, 'market_cap'):
                  total_mv = fast_info.market_cap or 0.0
             if high_52w == 0 and hasattr(fast_info, 'year_high'):
                  high_52w = fast_info.year_high or 0.0
             if low_52w == 0 and hasattr(fast_info, 'year_low'):
                  low_52w = fast_info.year_low or 0.0
             
             return RealtimeQuote(
                 code=stock_code,
                 name=name,
                 price=price,
                 change_pct=change_pct,
                 change_amount=change_amount,
                 volume_ratio=1.0, 
                 turnover_rate=0.0,
                 pe_ratio=pe,
                 pb_ratio=pb,
                 total_mv=total_mv,
                 circ_mv=total_mv,
                 high_52w=high_52w,
                 low_52w=low_52w
             )
        except Exception as e:
             logger.warning(f"Yfinance get_realtime_quote failed for {stock_code}: {e}")
             return None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从 Yahoo Finance 获取原始数据
        
        使用 yfinance.download() 获取历史数据
        
        流程：
        1. 转换股票代码格式
        2. 调用 yfinance API
        3. 处理返回数据
        """
        import yfinance as yf
        
        # 转换代码格式
        yf_code = self._convert_stock_code(stock_code)
        
        logger.debug(f"调用 yfinance.download({yf_code}, {start_date}, {end_date})")
        
        try:
            # 使用 yfinance 下载数据
            df = yf.download(
                tickers=yf_code,
                start=start_date,
                end=end_date,
                progress=False,  # 禁止进度条
                auto_adjust=True,  # 自动调整价格（复权）
            )
            
            if df.empty:
                raise DataFetchError(f"Yahoo Finance 未查询到 {stock_code} 的数据")
            
            return df
            
        except Exception as e:
            if isinstance(e, DataFetchError):
                raise
            raise DataFetchError(f"Yahoo Finance 获取数据失败: {e}") from e
    
    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        标准化 Yahoo Finance 数据
        
        yfinance 返回的列名：
        Open, High, Low, Close, Volume（索引是日期）
        
        需要映射到标准列名：
        date, open, high, low, close, volume, amount, pct_chg
        """
        df = df.copy()
        
        # 重置索引，将日期从索引变为列
        df = df.reset_index()
        
        # 列名映射（yfinance 使用首字母大写）
        column_mapping = {
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
        }
        
        df = df.rename(columns=column_mapping)
        
        # 计算涨跌幅（因为 yfinance 不直接提供）
        if 'close' in df.columns:
            df['pct_chg'] = df['close'].pct_change() * 100
            df['pct_chg'] = df['pct_chg'].fillna(0).round(2)
        
        # 计算成交额（yfinance 不提供，使用估算值）
        # 成交额 ≈ 成交量 * 平均价格
        if 'volume' in df.columns and 'close' in df.columns:
            df['amount'] = df['volume'] * df['close']
        else:
            df['amount'] = 0
        
        # 添加股票代码列
        df['code'] = stock_code
        
        # 只保留需要的列
        keep_cols = ['code'] + STANDARD_COLUMNS
        existing_cols = [col for col in keep_cols if col in df.columns]
        df = df[existing_cols]
        
        return df


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)
    
    fetcher = YfinanceFetcher()
    
    try:
        df = fetcher.get_daily_data('600519')  # 茅台
        print(f"获取成功，共 {len(df)} 条数据")
        print(df.tail())
    except Exception as e:
        print(f"获取失败: {e}")
