#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
 双均线交叉策略量化回测 —— 510300（沪深300ETF）
--------------------------------------------------------------------------------
 依赖清单（pip install）：
   pip install akshare matplotlib pandas numpy
================================================================================
 策略说明：
   - 标的：510300（沪深300ETF）
   - 周期：日K线，最近1年
   - 策略：MA10 上穿 MA30 → 金叉买入（全仓）
          MA10 下穿 MA30 → 死叉卖出（清仓）
   - 初始资金：100,000 元
   - 每次只持有一个仓位，全仓进出
 输出指标：
   - 累计收益率、年化收益率、最大回撤、夏普比率、交易次数
   - K线图 + 两条均线 + 买卖点标记
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import akshare as ak
import warnings
import os

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# 第0步：确定输出目录 —— 图表保存在脚本所在目录
# ──────────────────────────────────────────────────────────────────────────────
# 使用 __file__ 获取脚本位置，确保图表保存到正确目录
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # 如果在交互式环境运行，使用当前工作目录
    SCRIPT_DIR = os.getcwd()
CHART_PATH = os.path.join(SCRIPT_DIR, '双均线策略回测图表.png')

print(f"脚本目录: {SCRIPT_DIR}")
print(f"图表保存路径: {CHART_PATH}")

# ──────────────────────────────────────────────────────────────────────────────
# 第1步：获取数据 —— 使用 akshare 获取510300日K线
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("双均线交叉策略量化回测")
print("标的: 510300（沪深300ETF） | 初始资金: 100,000 元")
print("策略: MA10 × MA30 双均线交叉")
print("=" * 60)

print("\n[1/5] 正在获取数据...")

# 禁用系统代理（避免企业防火墙干扰）
os.environ['NO_PROXY'] = '*'

# 通过新浪接口获取510300历史日K线数据
# 返回列：date, open, high, low, close, volume, amount
df = ak.fund_etf_hist_sina(symbol='sh510300')

# 确保列名一致
df.rename(columns={
    'date': 'date',
    'open': 'open',
    'high': 'high',
    'low': 'low',
    'close': 'close',
    'volume': 'volume',
    'amount': 'amount'
}, inplace=True)

# 日期列转 datetime 类型，并设为索引
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# 按日期升序排列
df.sort_index(inplace=True)

# 筛选最近1年的数据
one_year_ago = pd.Timestamp.today() - pd.DateOffset(years=1)
df = df[df.index >= one_year_ago]

# 将价格列转为 float 类型
for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"    数据获取完成：{len(df)} 个交易日")
print(f"    时间范围：{df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")

# ──────────────────────────────────────────────────────────────────────────────
# 第2步：计算技术指标 —— 10日均线 和 30日均线
# ──────────────────────────────────────────────────────────────────────────────
print("\n[2/5] 计算均线指标...")

# MA10：最近10个交易日收盘价的简单移动平均
df['MA10'] = df['close'].rolling(window=10, min_periods=1).mean()

# MA30：最近30个交易日收盘价的简单移动平均
df['MA30'] = df['close'].rolling(window=30, min_periods=1).mean()

print(f"    MA10 计算完成（10日均线）")
print(f"    MA30 计算完成（30日均线）")

# ──────────────────────────────────────────────────────────────────────────────
# 第3步：逐日回测引擎 —— 模拟真实交易
# ──────────────────────────────────────────────────────────────────────────────
print("\n[3/5] 运行回测引擎...")

# 初始参数
initial_capital = 100000.0  # 初始资金（元）
capital = initial_capital    # 当前可用资金
position = 0                 # 当前持仓数量（股）
holding = False              # 是否持仓中
trade_records = []           # 交易记录列表
daily_net_value = []         # 每日净值序列

# ── 逐日遍历回测 ──
# 从第30个交易日开始（前29天MA30取不到足够数据，不构成有效信号）
for i in range(30, len(df)):
    today = df.iloc[i]        # 今日数据
    yesterday = df.iloc[i-1]  # 昨日数据
    today_date = df.index[i]  # 今日日期

    # 提取今日和昨日的均线值
    ma10_today = today['MA10']
    ma30_today = today['MA30']
    ma10_yest = yesterday['MA10']
    ma30_yest = yesterday['MA30']

    # 今日收盘价
    close_price = today['close']

    # ════════════════════════════════════════════
    # 判断是否出现【金叉】信号 → 买入
    # 金叉定义：昨日 MA10 ≤ MA30，今日 MA10 > MA30
    # 即 MA10 从下方上穿了 MA30 → 趋势转多
    # ════════════════════════════════════════════
    if not holding and ma10_yest <= ma30_yest and ma10_today > ma30_today:
        # 按整手（100股）买入，确保不超过可用资金
        buy_shares = int(capital // (close_price * 100)) * 100
        if buy_shares > 0:
            cost = buy_shares * close_price
            capital -= cost
            position = buy_shares
            holding = True
            trade_records.append({
                'date': today_date,
                'type': '买入',
                'price': round(close_price, 3),
                'shares': buy_shares,
                'amount': round(cost, 2),
                'capital_after': round(capital, 2)
            })

    # ════════════════════════════════════════════
    # 判断是否出现【死叉】信号 → 卖出
    # 死叉定义：昨日 MA10 ≥ MA30，今日 MA10 < MA30
    # 即 MA10 从上方下穿了 MA30 → 趋势转空
    # ════════════════════════════════════════════
    elif holding and ma10_yest >= ma30_yest and ma10_today < ma30_today:
        # 全仓卖出所有持股
        revenue = position * close_price
        capital += revenue
        trade_records.append({
            'date': today_date,
            'type': '卖出',
            'price': round(close_price, 3),
            'shares': position,
            'amount': round(revenue, 2),
            'capital_after': round(capital, 2)
        })
        position = 0
        holding = False

    # ── 计算当日总资产（净值） ──
    # 总资产 = 可用现金 + 持仓市值
    total_asset = capital + position * close_price
    daily_net_value.append({
        'date': today_date,
        'total_asset': round(total_asset, 2),
        'close': close_price,
        'holding': holding
    })

# ── 回测结束处理 ──
# 若在最后一天仍有持仓，按收盘价强制平仓
if holding:
    last_close = df.iloc[-1]['close']
    capital += position * last_close
    trade_records.append({
        'date': df.index[-1],
        'type': '卖出（强制平仓）',
        'price': round(last_close, 3),
        'shares': position,
        'amount': round(position * last_close, 2),
        'capital_after': round(capital, 2)
    })
    position = 0
    holding = False

# 将每日净值列表转为 DataFrame
net_value_df = pd.DataFrame(daily_net_value)
net_value_df.set_index('date', inplace=True)

# 将交易记录转为 DataFrame
trades_df = pd.DataFrame(trade_records)

print(f"    回测完成！共执行 {len(trades_df)} 笔交易（{len(trades_df)//2} 次买卖循环）")

# ──────────────────────────────────────────────────────────────────────────────
# 第4步：计算绩效指标
# ──────────────────────────────────────────────────────────────────────────────
print("\n[4/5] 计算绩效指标...")

# --- 4.1 累计收益率 ---
final_asset = net_value_df['total_asset'].iloc[-1]
total_return = (final_asset - initial_capital) / initial_capital * 100

# --- 4.2 年化收益率 ---
# 年化收益率 = (最终资产/初始资产)^(1/年数) - 1
trading_days = len(net_value_df)
years = trading_days / 252  # 252个交易日 ≈ 1年
annual_return = ((final_asset / initial_capital) ** (1 / years) - 1) * 100

# --- 4.3 最大回撤 ---
# 最大回撤 = 从历史最高点到最低点的最大跌幅
net_value_df['peak'] = net_value_df['total_asset'].cummax()           # 历史最高净值
net_value_df['drawdown'] = (net_value_df['total_asset'] - net_value_df['peak']) / net_value_df['peak'] * 100
max_drawdown = net_value_df['drawdown'].min()                          # 最大回撤

# --- 4.4 夏普比率 ---
# 夏普比率 = (年化收益率 - 无风险利率) / 年化波动率
net_value_df['daily_return'] = net_value_df['total_asset'].pct_change()
daily_returns = net_value_df['daily_return'].dropna()
annual_vol = daily_returns.std() * np.sqrt(252)
risk_free_rate = 0.02  # 假设无风险利率 2%
sharpe_ratio = (annual_return / 100 - risk_free_rate) / annual_vol if annual_vol > 0 else 0

# --- 4.5 交易次数 ---
trade_count = len(trades_df)

# ── 输出绩效报告 ──
print(f"\n{'=' * 50}")
print(f"📊 回测绩效报告")
print(f"{'=' * 50}")
print(f"  累计收益率：      {total_return:>8.2f}%")
print(f"  年化收益率：      {annual_return:>8.2f}%")
print(f"  最大回撤：        {max_drawdown:>8.2f}%")
print(f"  夏普比率：        {sharpe_ratio:>8.2f}")
print(f"  交易次数：        {trade_count:>8d} 次")
print(f"  最终资产：        {final_asset:>8,.2f} 元")
print(f"  交易天数：        {trading_days:>8d} 天")
print(f"{'=' * 50}")

# 打印交易明细
print(f"\n📝 交易明细：")
for _, trade in trades_df.iterrows():
    action_type = trade['type']
    print(f"  [{trade['date'].strftime('%Y-%m-%d')}] {action_type:　<6} "
          f"价格={trade['price']:>7.3f}  数量={trade['shares']:>5d}股  "
          f"金额={trade['amount']:>9.2f}  余额={trade['capital_after']:>9.2f}")

# ──────────────────────────────────────────────────────────────────────────────
# 第5步：绘制图表 —— 价格 + 均线 + 买卖点 + 净值曲线
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n[5/5] 绘制图表...")

# 设置中文字体（Windows系统优先使用微软雅黑）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建画布：上部分为价格走势，下部分为净值曲线
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle('510300（沪深300ETF）双均线策略回测 —— MA10 × MA30', fontsize=16, fontweight='bold')

# ═══════════════════════════════════════════════
# 上图：价格走势 + 均线 + 买卖点
# ═══════════════════════════════════════════════

# 绘制收盘价曲线（黑线）
ax1.plot(df.index, df['close'], label='收盘价', color='#333333', linewidth=1, alpha=0.7)

# 绘制MA10均线（蓝线，短期均线，反应灵敏）
ax1.plot(df.index, df['MA10'], label='MA10（10日均线）', color='#2196F3', linewidth=1.5)

# 绘制MA30均线（红线，长期均线，趋势更稳定）
ax1.plot(df.index, df['MA30'], label='MA30（30日均线）', color='#F44336', linewidth=1.5)

# 标记金叉买入点（绿色上箭头 ▲）
if not trades_df.empty:
    buy_trades = trades_df[trades_df['type'].str.contains('买入')]
    if not buy_trades.empty:
        ax1.scatter(buy_trades['date'], buy_trades['price'],
                    marker='^', color='#4CAF50', s=150, label='买入（金叉信号）',
                    zorder=5, edgecolors='darkgreen', linewidth=1)

    # 标记死叉卖出点（红色下箭头 ▼）
    sell_trades = trades_df[trades_df['type'].str.contains('卖出')]
    if not sell_trades.empty:
        ax1.scatter(sell_trades['date'], sell_trades['price'],
                    marker='v', color='#FF5722', s=150, label='卖出（死叉信号）',
                    zorder=5, edgecolors='darkred', linewidth=1)

ax1.set_ylabel('价格（元）', fontsize=11)
ax1.legend(loc='best', fontsize=10, framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.set_title('价格走势与均线信号', fontsize=13, fontweight='bold')

# ═══════════════════════════════════════════════
# 下图：账户净值曲线
# ═══════════════════════════════════════════════

# 绘制净值曲线（绿色线）
ax2.plot(net_value_df.index, net_value_df['total_asset'],
         label='账户净值', color='#4CAF50', linewidth=1.5)

# 填充盈利区域（净值高于初始资金，浅绿色）
ax2.fill_between(net_value_df.index, initial_capital, net_value_df['total_asset'],
                 where=net_value_df['total_asset'] >= initial_capital,
                 color='green', alpha=0.08, label='盈利区域')

# 填充亏损区域（净值低于初始资金，浅红色）
ax2.fill_between(net_value_df.index, initial_capital, net_value_df['total_asset'],
                 where=net_value_df['total_asset'] < initial_capital,
                 color='red', alpha=0.08, label='亏损区域')

# 标注初始资金线（灰色虚线）
ax2.axhline(y=initial_capital, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax2.text(net_value_df.index[0], initial_capital * 1.02,
         f'初始资金 ¥{initial_capital:,.0f}', fontsize=9, color='gray')

ax2.set_ylabel('账户净值（元）', fontsize=11)
ax2.set_xlabel('日期', fontsize=11)
ax2.legend(loc='best', fontsize=10, framealpha=0.9)
ax2.grid(True, alpha=0.3)
ax2.set_title('账户净值曲线', fontsize=13, fontweight='bold')

# 统一设置日期格式（显示 年-月）
for ax in [ax1, ax2]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()

# ── 保存图表 ──
# 尝试保存到脚本所在目录，如果失败则保存到当前工作目录
try:
    plt.savefig(CHART_PATH, dpi=150, bbox_inches='tight')
    print(f"    图表已保存: {CHART_PATH}")
except Exception:
    # 备用：保存到当前工作目录
    fallback_path = os.path.join(os.getcwd(), '双均线策略回测图表.png')
    plt.savefig(fallback_path, dpi=150, bbox_inches='tight')
    print(f"    图表已保存（备用路径）: {fallback_path}")

# ── 显示图表（如在不支持GUI的环境中会自动跳过）──
try:
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()
except Exception:
    pass

print("\n✅ 回测完成！")
