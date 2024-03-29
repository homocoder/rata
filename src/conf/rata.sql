select
    tstamp, symbol, interval,
    open, high, low, close, volume,
    momentum_rsi,
    momentum_stoch_rsi,
    trend_macd,
    trend_cci,
    "open_SROC_6",
    "open_SROC_9",
    "open_SROC_15",
    "open_SROC_30",
    "open_SROC_60",
    "open_SROC_90",
    "high_SROC_6",
    "high_SROC_9",
    "high_SROC_15",
    "high_SROC_30",
    "high_SROC_60",
    "high_SROC_90",
    "low_SROC_6",
    "low_SROC_9",
    "low_SROC_15",
    "low_SROC_30",
    "low_SROC_60",
    "low_SROC_90",
    "close_SROC_6",
    "close_SROC_9",
    "close_SROC_15",
    "close_SROC_30",
    "close_SROC_60",
    "close_SROC_90",
    "volume_SROC_6",
    "volume_SROC_9",
    "volume_SROC_15",
    "volume_SROC_30",
    "volume_SROC_60",
    "volume_SROC_90",

    momentum_stoch_rsi_k,
    momentum_stoch_rsi_d,
    "momentum_rsi_SROC_6",
    "momentum_rsi_SROC_9",
    "momentum_rsi_SROC_15",
    "momentum_rsi_SROC_30",
    "momentum_rsi_SROC_60",
    "momentum_rsi_SROC_90",
    "momentum_stoch_rsi_SROC_6",
    "momentum_stoch_rsi_SROC_9",
    "momentum_stoch_rsi_SROC_15",
    "momentum_stoch_rsi_SROC_30",
    "momentum_stoch_rsi_SROC_60",
    "momentum_stoch_rsi_SROC_90",
    "momentum_stoch_rsi_k_SROC_6",
    "momentum_stoch_rsi_k_SROC_9",
    "momentum_stoch_rsi_k_SROC_15",
    "momentum_stoch_rsi_k_SROC_30",
    "momentum_stoch_rsi_k_SROC_60",
    "momentum_stoch_rsi_k_SROC_90",
    "momentum_stoch_rsi_d_SROC_6",
    "momentum_stoch_rsi_d_SROC_9",
    "momentum_stoch_rsi_d_SROC_15",
    "momentum_stoch_rsi_d_SROC_30",
    "momentum_stoch_rsi_d_SROC_60",
    "momentum_stoch_rsi_d_SROC_90",

    trend_macd_signal,
    trend_macd_diff,
    "trend_macd_SROC_6",
    "trend_macd_SROC_9",
    "trend_macd_SROC_15",
    "trend_macd_SROC_30",
    "trend_macd_SROC_60",
    "trend_macd_SROC_90",
    "trend_macd_signal_SROC_6",
    "trend_macd_signal_SROC_9",
    "trend_macd_signal_SROC_15",
    "trend_macd_signal_SROC_30",
    "trend_macd_signal_SROC_60",
    "trend_macd_signal_SROC_90",
    "trend_macd_diff_SROC_6",
    "trend_macd_diff_SROC_9",
    "trend_macd_diff_SROC_15",
    "trend_macd_diff_SROC_30",
    "trend_macd_diff_SROC_60",
    "trend_macd_diff_SROC_90",

    "trend_cci_SROC_6",
    "trend_cci_SROC_9",
    "trend_cci_SROC_15",
    "trend_cci_SROC_30",
    "trend_cci_SROC_60",
    "trend_cci_SROC_90",

    "y_close_SROC_6_shift-6",
    "y_B_close_SROC_6_shift-6",
    "y_S_close_SROC_6_shift-6",
    "y_close_SROC_9_shift-9",
    "y_B_close_SROC_9_shift-9",
    "y_S_close_SROC_9_shift-9",
    "y_close_SROC_15_shift-15",
    "y_B_close_SROC_15_shift-15",
    "y_S_close_SROC_15_shift-15",
    "y_close_SROC_30_shift-30",
    "y_B_close_SROC_30_shift-30",
    "y_S_close_SROC_30_shift-30",
    "y_close_SROC_60_shift-60",
    "y_B_close_SROC_60_shift-60",
    "y_S_close_SROC_60_shift-60",
    "y_close_SROC_90_shift-90",
    "y_B_close_SROC_90_shift-90",
    "y_S_close_SROC_90_shift-90"

from feateng
where interval=3
order by tstamp desc
limit 50000