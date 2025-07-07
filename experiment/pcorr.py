import pandas as pd
import numpy as np

# 데이터프레임 df가 있다고 가정하고, 'A'와 'B'는 두 컬럼의 이름입니다.
p_values = np.arange(0, 1.001, 0.001)
results = []  # 각 p 값에 대한 상관계수를 저장할 리스트입니다.

data = pd.read_excel('./hyperparam.xlsx')

for p in p_values:
    data['CoSE'] = data['Smatch#']*p + data['Simplehash-3']*(1-p)
    correlation_gen = data['CoSE'].corr(data['GENERAL'])
    correlation_exp = data['CoSE'].corr(data['EXP'])
    results.append((p, correlation_gen, correlation_exp))

# 이제 p 값과 그에 해당하는 상관계수를 출력합니다.
for p, corr_gen, corr_exp in results:
    print(f"p: {p:.3f}, Correlation with GENERAL: {corr_gen:.8f}, Correlation with EXP: {corr_exp:.6f}")

# 각 상관계수의 최대값과 그에 해당하는 p 값을 찾습니다.
max_corr_gen = max(results, key=lambda x: x[1])
max_corr_exp = max(results, key=lambda x: x[2])

print(f"\nMax correlation with GENERAL is {max_corr_gen[1]:.8f} at p = {max_corr_gen[0]:.3f}")
print(f"Max correlation with EXP is {max_corr_exp[2]:.8f} at p = {max_corr_exp[0]:.3f}")
