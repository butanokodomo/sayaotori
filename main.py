import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
import pandas_market_calendars as mcal
#-条件------------------------------------------------------------------------------------------------------------------------------------
st.title('サヤトリ')
#uploaded_file = st.file_uploader("Excelファイルをアップロード", type=["xlsx", "xls"])
# onedrive上のエクセルファイルのURL
exceldate = "https://1drv.ms/x/s!AgiIYO8iopUihSE_ZtAc3PrNIfd7?e=A929CH"
# エクセルファイルをPandasのDataFrameに読み込む
df = pd.read_excel(exceldate, engine='xlrd')
c1, c2 = st.columns([0.2, 0.2])
if df is not None:
    code1 = c1.selectbox('主', df['コード1'])
    code2 = c2.selectbox('脇', df['コード2'])
else:
    code1 = c1.text_input('主')
    code2 = c2.text_input('脇')
c3, c4 = st.columns([0.2, 0.2]) 
hyouzi = c3.selectbox('グラフの表示日数', (365, 720, 1095))
kabusa = c4.selectbox('株数誤差％', (10, 5))
c5 = st.columns([0.2, 0.2]) 
with c5[0]:syuuekikeisan = st.checkbox('total収益計算')

if code1 and code2:
    stock1 = yf.Ticker(f'{code1}.T')
    stock2 = yf.Ticker(f'{code2}.T')
else:st.warning('主と脇の株価コードを入力してください。')
# 営業日カレンダーを取得
jpex = mcal.get_calendar('JPX')
# 365日前の日付を計算
one_year_ago = dt.datetime.today() - dt.timedelta(days=hyouzi+120)
# 365日前から今日までの営業日を取得
valid_days = jpex.valid_days(start_date=one_year_ago, end_date=dt.datetime.today())
# 最初の営業日を取得
s = valid_days[0]
s2 = valid_days[75]
#移動平均日数④
days = 75
#-計算------------------------------------------------------------------------------------------------------------------------------------
# DataFrameを作成する
df1 = stock1.history(start=s)
df2 = stock2.history(start=s)
df3 = stock1.history(start=s2)
df4 = stock2.history(start=s2)
#購入日の始値取得
open1 = df1['Open'].iloc[-1]
open2 = df2['Open'].iloc[-1]
# 直近の営業日 終値取得
last1 = df1.tail(1)  # データフレームの最後の行を取得
last2 = df2.tail(1)  # データフレームの最後の行を取得
close1 = df1['Close'].iloc[-1]
close2 = df2['Close'].iloc[-1]
closebefore1 = df1['Close'].iloc[-2]
closebefore2 = df2['Close'].iloc[-2]
# サヤ比の計算
saya_ratio = round(df1['Close'] / df2['Close'], 3)
# 移動平均線の計算
sma = saya_ratio.rolling(window=days).mean()
# 75日移動平均の計算
saya_ma = round(saya_ratio.rolling(window=days).mean(), 3)
# ボリンジャーバンドの計算
std = saya_ratio.rolling(window=days, min_periods=days-1).std()
upper_band2 = sma + std*2
upper_band1 = sma + std
lower_band2 = sma - std*2
lower_band1 = sma - std

# サヤ比が移動平均に交差した回数をカウントする関数
def count_crossovers(saya_ratio, sma):
    count = 0
    for i in range(1, len(saya_ratio)):
        if (saya_ratio[i-1] < saya_ma[i-1]) and (saya_ratio[i] >= saya_ma[i]):
            count += 1
        elif (saya_ratio[i-1] > saya_ma[i-1]) and (saya_ratio[i] <= saya_ma[i]):
            count += 1
    return count
crossover_count = count_crossovers(saya_ratio, sma)

#相関関数
correlation = round(df3['Close'].corr(df4['Close']), 3)

# 乖離率の計算
distance_from_ma =round( (saya_ratio - saya_ma) / saya_ma, 3)
# σ値の計算
sigma_value = round((saya_ratio - saya_ma) / (upper_band2- saya_ma) * 2, 3)

#株数算出
def find_closest_value(row, matrix):
    closest_value = min(matrix, key=lambda x: abs(row - x))
    return closest_value

if open1 > open2:
    a, b = open2, open1
elif open1 < open2:
    a, b = open1, open2
    
data = []
for i in range(1, 12):    
    num = 100 * i
    data.append([num, a * num, b * num])

matrix = np.array(data)

row2 = matrix[:, 1]
row3 = matrix[:, 2]
row4 = np.array([find_closest_value(val, row3) for val in row2])

df = pd.DataFrame(matrix, columns=['株数', 'a', 'b'])
df['c'] = row4
df['差率'] = (df['c'] - df['a']) / df['a'] * 100

fi = df[(df['差率'] >= -kabusa) & (df['差率'] <= kabusa)]#誤差5%以内
fir = fi.index[0]+1
ra = int(df.loc[fir-1, 'c'] / df.loc[0, 'c'] * 100)


kabusu1, kabusu2 = (fir * 100, ra) if open1 == a else (ra, fir * 100)

# 収益を計算
if syuuekikeisan:
    buy1 = st.text_input('価格1')
    buy2 = st.text_input('価格2')
    
    syueki = (float(buy1) - close1) * kabusu1 + (close2 - float(buy2)) * kabusu2 if saya_ratio.mean() > saya_ma.mean() else (float(buy2) - close2) * kabusu2 + (close1 - float(buy1)) * kabusu1
    
    st.write(f'収益：{syueki:,.0f}円')

#当日変動価格
# 当日の終値から前日の終値を引いて変動を計算し、hendou列を追加する
hendou1 = df1['Close'].diff() * kabusu1 if saya_ratio.mean() < saya_ma.mean() else df1['Close'].diff() * kabusu1*-1
hendou2 = df2['Close'].diff() * kabusu2*-1 if saya_ratio.mean() < saya_ma.mean() else df2['Close'].diff() * kabusu2

# 変動の合計を計算し、totalhendouに代入する
totalhendou = hendou1+hendou2

#hendou1 = (close1 - closebefore1)*kabusu1 if saya_ratio.mean() < saya_ma.mean() else (closebefore1-close1)*kabusu1
#hendou2 = (closebefore2-close2)*kabusu2 if saya_ratio.mean() < saya_ma.mean() else (close2 - closebefore2)*kabusu2
#totalhendou = (hendou1+hendou2)
#-印字------------------------------------------------------------------------------------------------------------------------------------
st.write(f'{code1}', f'　{kabusu1:,.0f}株')#f'　{hendou1:,.0f}円')
st.write(f'{code2}', f'　{kabusu2:,.0f}株')

table = pd.DataFrame({
    "相関": correlation,
    "σ値": sigma_value,
    "サヤ比": saya_ratio,
    "移動平均": saya_ma,
    "乖離率": (distance_from_ma * 100).round(3).astype(str) + "%",
    "交差数": crossover_count,
    "差益" : totalhendou
    
})
# 日付のみを抽出して行名に設定
table.index = df1.index.strftime('%m/%d')
# 列名を1行目に移動
table.columns.name = "日付"
# 2行目のDateを削除
table.index.name = ""

# テーブルの表示
st.write(table.tail(5)) #表示する行数を入れる

import streamlit as st

# グラフの描画
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(saya_ratio, color='#0000ff', alpha=0.9, label='Saya Ratio')
ax.plot(sma, color='#ff3300', alpha=0.9, label='SMA ({})'.format(days))
ax.fill_between(saya_ratio.index, upper_band1, lower_band1, color="#33ff66", alpha=0.7, label="$1\sigma$")
ax.fill_between(saya_ratio.index, upper_band2, lower_band2, color="#33ff66", alpha=0.3, label="$2\sigma$")
ax.legend()
ax.set_xlim(saya_ratio.index[75], saya_ratio.index[-1])  # 75日目から表示

# x軸のメジャー目盛を2週間ごとに設定する
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

# x軸の目盛の表示形式を設定する
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))

# x軸の目盛ラベルを縦に斜めに表示する
plt.xticks(rotation=75, ha='right')  # 修正: 数値の指定は不要です


# Streamlitでグラフを表示
 # Streamlitでグラフを表示
st.pyplot(fig)
