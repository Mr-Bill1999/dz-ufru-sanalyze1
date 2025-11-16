import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import time

np.random.seed(123)

class BrandImpactEvaluator:
    def __init__(self, alpha=0.05, target_power=0.8):
        self.alpha = alpha
        self.target_power = target_power

    @staticmethod
    def z_test_one_sided_vec(cnt_tests, cnt_ctrls, N):
        """Односторонний z-test для векторных данных"""
        p_tests = cnt_tests / N
        p_ctrls = cnt_ctrls / N
        p_pool = (cnt_tests + cnt_ctrls) / (2*N)
        se = np.sqrt(p_pool * (1 - p_pool) * (2.0/N))
        z = np.zeros_like(se)
        valid = se > 0
        z[valid] = (p_tests[valid] - p_ctrls[valid]) / se[valid]
        pvals = 1 - norm.cdf(z)
        return pvals

    @staticmethod
    def bootstrap_ratio_pval(cnt_test, N, cnt_ctrl, Nctrl, n_boot=400):
        """Быстрый bootstrap для ratio (p_test / p_control)"""
        p_test = cnt_test / N
        p_ctrl = cnt_ctrl / Nctrl
        boots_t = np.random.binomial(N, p_test, size=n_boot) / N
        boots_c = np.random.binomial(Nctrl, p_ctrl, size=n_boot) / Nctrl
        ratio = np.where(boots_c == 0, np.inf, boots_t / boots_c)
        return np.mean(ratio <= 1.0)

    def simulate_uplift(self, N, method='z',
                        M=300, K=300,
                        uplift_grid=np.linspace(0.0, 0.2, 101),
                        p_control_low=0.2, p_control_high=0.8,
                        n_boot=200):
        """Симуляция минимального detectable uplift для заданного N"""
        mins = np.full(M, np.nan)
        for i in range(M):
            p0 = np.random.uniform(p_control_low, p_control_high)
            for u in uplift_grid:
                p_test = min(p0*(1+u), 0.999)
                cnt_ctrls = np.random.binomial(N, p0, size=K)
                cnt_tests = np.random.binomial(N, p_test, size=K)

                if method == 'z':
                    pvals = self.z_test_one_sided_vec(cnt_tests, cnt_ctrls, N)
                elif method == 'bootstrap':
                    pvals = np.array([self.bootstrap_ratio_pval(cnt_tests[k], N, cnt_ctrls[k], N, n_boot=n_boot)
                                      for k in range(K)])
                else:
                    raise ValueError("method must be 'z' or 'bootstrap'")

                power = np.mean(pvals < self.alpha)
                if power >= self.target_power:
                    mins[i] = u
                    break
        return mins

    def run_all_sample_sizes(self, Ns, method='z', M=None, K=None, n_boot=200, verbose=True):
        """Симуляция для всех размеров выборки и формирование итоговой таблицы"""
        out = []
        if M is None: M = 300 if method=='z' else 120
        if K is None: K = 300 if method=='z' else 150
        uplift_grid = np.linspace(0.0, 0.2, 101) if method=='z' else np.linspace(0.0, 0.2, 51)

        for N in Ns:
            t0 = time.time()
            mins = self.simulate_uplift(N, method=method, M=M, K=K, uplift_grid=uplift_grid, n_boot=n_boot)
            t1 = time.time()
            mins_clean = mins[~np.isnan(mins)]
            if len(mins_clean) == 0:
                p25 = p50 = p75 = np.nan
            else:
                p25 = np.percentile(mins_clean, 25)
                p50 = np.percentile(mins_clean, 50)
                p75 = np.percentile(mins_clean, 75)

            out.append({
                'N': N, 'method': method,
                'p25_pct': p25*100, 'median_pct': p50*100, 'p75_pct': p75*100,
                'found_frac': len(mins_clean)/len(mins)
            })
            if verbose:
                print(f"N={N} done; time={(t1-t0):.1f}s, detected_frac={len(mins_clean)/len(mins):.2f}")
        return pd.DataFrame(out)

    def run_demo(self, N=5000, baseline_awareness=0.3, effect_size=0.05, n_boot=500):
        """Демонстрация работы метода на одной выборке"""
        p_test = min(baseline_awareness + effect_size, 0.99)
        control_data = np.random.binomial(1, baseline_awareness, N)
        test_data = np.random.binomial(1, p_test, N)
        control_rate = np.mean(control_data)
        test_rate = np.mean(test_data)
        relative_effect = test_rate / control_rate - 1

        # z-test
        pval_z = self.z_test_one_sided_vec(np.array([test_data.sum()]), np.array([control_data.sum()]), N)[0]

        # bootstrap ratio
        pval_boot = self.bootstrap_ratio_pval(test_data.sum(), N, control_data.sum(), N, n_boot=n_boot)

        print("=== DEMO ===")
        print(f"Контрольная группа: {control_rate:.1%}")
        print(f"Тестовая группа: {test_rate:.1%}")
        print(f"Относительный эффект: {relative_effect:.1%}")
        print(f"z-test p-value: {pval_z:.4f}")
        print(f"bootstrap p-value: {pval_boot:.4f}")

        return {
            'control_rate': control_rate,
            'test_rate': test_rate,
            'relative_effect': relative_effect,
            'z_pval': pval_z,
            'bootstrap_pval': pval_boot
        }

if __name__ == '__main__':
    Ns = [1000, 2000, 3000, 5000, 10000, 20000]
    experiment = BrandImpactEvaluator()

    # Запуск симуляций
    df_z = experiment.run_all_sample_sizes(Ns, method='z')
    df_boot = experiment.run_all_sample_sizes(Ns, method='bootstrap', n_boot=200)

    df_res = pd.concat([df_z, df_boot], ignore_index=True)
    print(df_res)

    # Сохраняем CSV
    df_res.to_csv("detectable_uplifts_combined.csv", index=False)

    # Демонстрация работы на одной выборке
    demo = experiment.run_demo(N=5000, baseline_awareness=0.3, effect_size=0.05, n_boot=500)

    # График мощности
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    for method, df_method in df_res.groupby('method'):
        for N in Ns:
            df_plot = df_method[df_method['N']==N]
            plt.plot([df_plot['p25_pct'].values[0], df_plot['p75_pct'].values[0]], [N, N],
                     marker='o', label=f'{method}, N={N}')
    plt.xlabel('Detectable uplift (%)')
    plt.ylabel('Sample size (N)')
    plt.title('Диапазон детектируемого relative uplift (25–75 перцентиль)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
