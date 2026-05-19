# coding=utf-8
"""
=============================================================================
 双均线交叉策略 —— 东方财富量化终端 (EMQuant) 版
-----------------------------------------------------------------------------
 标的：510300（沪深300ETF）
 策略：MA10 上穿 MA30 → 金叉买入（全仓）
       MA10 下穿 MA30 → 死叉卖出（清仓）
 运行模式：定时任务，每天 14:50 执行一次
-----------------------------------------------------------------------------
 使用方法：
   1. 打开「东方财富量化终端」→ 量化 → 我的策略 → 新建策略
   2. 语言选择 Python，将此文件内容粘贴到 main.py
   3. 先选「仿真交易」跑通，确认无问题后再切「实盘交易」
=============================================================================
"""
from __future__ import print_function, absolute_import
from gm.api import *
import numpy as np


# ============================================================================
# 策略参数（可在此修改）
# ============================================================================
SYMBOL = 'SHSE.510300'     # 交易标的：510300 沪深300ETF（上海交易所）
MA_SHORT = 10              # 短期均线周期（交易日）
MA_LONG = 30               # 长期均线周期（交易日）
CHECK_TIME = '14:50:00'    # 每天检查信号的时间（收盘前10分钟）
HISTORY_DAYS = 50          # 取多少天历史数据用于计算均线（>= MA_LONG + 10）


# ============================================================================
# 初始化 —— 设置定时任务
# ============================================================================
def init(context):
    """
    策略初始化：设置每天定时执行的时间点
    """
    # 每天在 CHECK_TIME 执行 algo 函数
    schedule(schedule_func=algo, date_rule='1d', time_rule=CHECK_TIME)
    
    # 打印启动信息
    log(level='info', msg='双均线策略已启动 | 标的: {} | MA{}×MA{} | 检查时间: {}'.format(
        SYMBOL, MA_SHORT, MA_LONG, CHECK_TIME), source='strategy')
    print('[双均线策略] 初始化完成，每天 {} 执行信号检查'.format(CHECK_TIME))


# ============================================================================
# 核心逻辑 —— 每天定时执行
# ============================================================================
def algo(context):
    """
    每日执行的交易逻辑：
    1. 取最近 HISTORY_DAYS 天的日K线
    2. 计算 MA10 和 MA30
    3. 判断金叉/死叉
    4. 执行买卖
    """
    # ── 步骤1：获取历史数据 ──
    try:
        # history() 取日线数据，fields 指定需要的字段
        data = history(
            symbol=SYMBOL,
            frequency='1d',
            start_time=context.now - timedelta(days=HISTORY_DAYS + 20),  # 多取一些确保够
            end_time=context.now,
            fields='close',
            adjust=ADJUST_PREV,    # 前复权
            df=True
        )
    except Exception as e:
        log(level='error', msg='获取历史数据失败: {}'.format(str(e)), source='strategy')
        print('[错误] 获取历史数据失败: {}'.format(e))
        return

    # 检查数据是否足够
    if data is None or len(data) < MA_LONG + 5:
        log(level='warning', msg='历史数据不足，跳过本次检查（现有 {} 条）'.format(
            len(data) if data is not None else 0), source='strategy')
        print('[警告] 历史数据不足（需要至少 {} 条），跳过'.format(MA_LONG + 5))
        return

    # ── 步骤2：计算均线 ──
    data['MA_SHORT'] = data['close'].rolling(window=MA_SHORT).mean()
    data['MA_LONG'] = data['close'].rolling(window=MA_LONG).mean()

    # 取最近两天的均线值
    latest = data.iloc[-1]
    previous = data.iloc[-2]

    ma_short_today = latest['MA_SHORT']
    ma_long_today = latest['MA_LONG']
    ma_short_yest = previous['MA_SHORT']
    ma_long_yest = previous['MA_LONG']

    # 数据有效性检查
    if np.isnan(ma_short_yest) or np.isnan(ma_long_yest):
        print('[警告] 均线数据不足（存在NaN），跳过本次检查')
        return

    close_price = latest['close']

    # ── 步骤3：获取当前账户状态 ──
    account = context.account()
    cash_info = account.cash
    available_cash = cash_info.available if hasattr(cash_info, 'available') else cash_info.nav

    # 查询当前持仓
    pos = account.position(symbol=SYMBOL, side=PositionSide_Long)
    holding_volume = pos.volume if pos else 0  # 当前持仓数量（股）

    # ── 步骤4：判断金叉/死叉 ──
    # 金叉：昨日 MA_SHORT <= MA_LONG，今日 MA_SHORT > MA_LONG
    is_golden_cross = (ma_short_yest <= ma_long_yest) and (ma_short_today > ma_long_today)

    # 死叉：昨日 MA_SHORT >= MA_LONG，今日 MA_SHORT > MA_LONG 不成立，且 MA_SHORT < MA_LONG
    is_death_cross = (ma_short_yest >= ma_long_yest) and (ma_short_today < ma_long_today)

    # ── 步骤5：执行交易 ──
    if is_golden_cross and holding_volume == 0:
        # ─── 金叉买入信号 ───
        # 计算可买入的整手数量（ETF 100股/手）
        lots = int(available_cash / (close_price * 100))
        buy_volume = lots * 100

        if buy_volume > 0:
            # 限价单：以当前收盘价附近的价格委托
            order_volume(
                symbol=SYMBOL,
                volume=buy_volume,
                side=OrderSide_Buy,
                order_type=OrderType_Limit,
                position_effect=PositionEffect_Open,
                price=close_price
            )
            msg = ('🔵 金叉买入信号 | 价格: {:.3f} | 数量: {}股 | '
                   '金额: {:.2f}元 | MA_SHORT: {:.3f} | MA_LONG: {:.3f}').format(
                close_price, buy_volume, buy_volume * close_price,
                ma_short_today, ma_long_today)
            log(level='info', msg=msg, source='strategy')
            print(msg)
        else:
            msg = '⚠️ 金叉信号触发，但可用资金不足买入1手（可用: {:.2f}元, 需: {:.2f}元）'.format(
                available_cash, close_price * 100)
            log(level='warning', msg=msg, source='strategy')
            print(msg)

    elif is_death_cross and holding_volume > 0:
        # ─── 死叉卖出信号 ───
        order_volume(
            symbol=SYMBOL,
            volume=holding_volume,
            side=OrderSide_Sell,
            order_type=OrderType_Limit,
            position_effect=PositionEffect_Close,
            price=close_price
        )
        msg = ('🔴 死叉卖出信号 | 价格: {:.3f} | 数量: {}股 | '
               '金额: {:.2f}元 | MA_SHORT: {:.3f} | MA_LONG: {:.3f}').format(
            close_price, holding_volume, holding_volume * close_price,
            ma_short_today, ma_long_today)
        log(level='info', msg=msg, source='strategy')
        print(msg)

    else:
        # ─── 无信号，持有不动 ───
        status = '持仓中' if holding_volume > 0 else '空仓等待'
        print('[{}] {} | MA_SHORT={:.3f} | MA_LONG={:.3f} | 价格={:.3f} | 持仓={}股 | 现金={:.2f}'.format(
            context.now.strftime('%Y-%m-%d %H:%M'), status,
            ma_short_today, ma_long_today, close_price,
            holding_volume, available_cash))


# ============================================================================
# 交易事件回调 —— 监控订单状态
# ============================================================================
def on_order_status(context, order):
    """
    订单状态变化时触发，用于监控成交情况
    """
    status_map = {
        OrderStatus_New: '新订单',
        OrderStatus_PartiallyFilled: '部分成交',
        OrderStatus_Filled: '全部成交',
        OrderStatus_Canceled: '已撤销',
        OrderStatus_PendingCancel: '待撤销',
        OrderStatus_Rejected: '已拒绝',
    }
    status_text = status_map.get(order.status, '未知状态({})'.format(order.status))
    
    if order.status == OrderStatus_Filled:
        print('✅ 订单成交 | {} | {} | 价格: {:.3f} | 数量: {}股 | 金额: {:.2f}'.format(
            order.symbol, status_text, order.price, order.volume, order.volume * order.price))
    elif order.status == OrderStatus_Rejected:
        print('❌ 订单被拒绝 | {} | 原因: {}'.format(order.symbol, order.reject_reason))
    else:
        print('📋 订单更新 | {} | {} | 价格: {:.3f} | 数量: {}'.format(
            order.symbol, status_text, order.price, order.volume))


def on_execution_report(context, exec_rpt):
    """
    成交回报事件
    """
    print('📊 成交回报 | {} | {}股 @ {:.3f}'.format(exec_rpt.symbol, exec_rpt.volume, exec_rpt.price))


# ============================================================================
# 账户状态事件
# ============================================================================
def on_account_status(context, account):
    """
    账户状态变化时触发
    """
    cash = account.cash
    available = cash.available if hasattr(cash, 'available') else cash.nav
    print('[账户状态] 可用资金: {:.2f}元 | 总资产: {:.2f}元'.format(available, cash.nav))


# ============================================================================
# 错误处理
# ============================================================================
def on_error(context, code, msg):
    """
    策略运行错误时触发
    """
    print('[策略错误] 错误码: {} | 信息: {}'.format(code, msg))
    log(level='error', msg='错误码: {} | 信息: {}'.format(code, msg), source='strategy')


# ============================================================================
# 回测结束回调（仅在回测模式触发）
# ============================================================================
def on_backtest_finished(context, indicator):
    """
    回测结束时输出绩效指标
    """
    print('\n' + '=' * 50)
    print('📊 回测绩效报告')
    print('=' * 50)
    print('  累计收益率:     {:.2f}%'.format(indicator.total_return * 100))
    print('  年化收益率:     {:.2f}%'.format(indicator.annual_return * 100))
    print('  最大回撤:       {:.2f}%'.format(indicator.max_drawdown * 100))
    print('  夏普比率:       {:.2f}'.format(indicator.sharpe_ratio))
    print('  胜率:           {:.2f}%'.format(indicator.win_ratio * 100))
    print('  交易次数:       {}'.format(indicator.trade_count))
    print('=' * 50)


# ============================================================================
# 策略入口
# ============================================================================
if __name__ == '__main__':
    """
    run() 函数的 mode 参数：
      - MODE_BACKTEST: 回测模式（用历史数据验证策略）
      - MODE_LIVE:     实时模式（仿真交易 or 实盘交易）
    
    使用步骤：
      1. 先在终端内回测：mode=MODE_BACKTEST
      2. 回测满意后 → 仿真交易：mode=MODE_LIVE（在终端选择仿真账户）
      3. 仿真跑稳后   → 实盘交易：mode=MODE_LIVE（在终端选择实盘账户）
    """

    # ⚠️ 请替换为你在东方财富量化终端中获取的真实 ID
    run(
        strategy_id='your_strategy_id_here',      # ← 在终端「我的策略」中查看
        filename='main.py',
        mode=MODE_LIVE,                            # 实时模式（终端会自动切换）
        token='your_token_id_here',                # ← 在终端「用户管理 → 密钥管理」中查看
        backtest_start_time='2025-05-01 08:00:00',
        backtest_end_time='2026-04-30 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=5000,                # 回测初始资金（对应你的5k）
        backtest_commission_ratio=0.00025,         # 佣金万2.5
        backtest_slippage_ratio=0.0001,            # 滑点万1
    )
