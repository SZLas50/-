import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 假设数据集已加载为 df，包含列：age, gender, bill_amount, repayment_record, repayment_date, default
# 这里用示例数据代替，实际使用时替换为你的数据
# df = pd.read_csv('your_data.csv')

# 示例数据（请替换为实际数据）
data = {
    'age': [25, 30, 35, 40, 45, 50, 55, 60],
    'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
    'bill_amount': [1000, 2000, 1500, 3000, 2500, 1800, 2200, 2800],
    'repayment_record': [1, 0, 1, 0, 1, 0, 1, 0],  # 假设1为按时还款，0为未按时
    'repayment_date': pd.date_range('2023-01-01', periods=8),
    'default': [0, 1, 0, 1, 0, 1, 0, 1]  # 0: 不违约, 1: 违约
}
df = pd.DataFrame(data)

# 设置中文字体（如果需要显示中文标签）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False

# 创建子图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('信贷违约数据可视化分析', fontsize=16)

# 1. 目标变量的类别分布图（违约vs不违约）
default_counts = df['default'].value_counts()
axes[0, 0].pie(default_counts, labels=['不违约', '违约'], autopct='%1.1f%%', startangle=90)
axes[0, 0].set_title('目标变量类别分布')
axes[0, 0].axis('equal')  # 使饼图为圆形

# 2. 特征之间的相关性热力图（选择数值特征）
# 假设数值特征包括 age, bill_amount, repayment_record
numeric_features = ['age', 'bill_amount', 'repayment_record']
corr_matrix = df[numeric_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[0, 1])
axes[0, 1].set_title('特征相关性热力图')

# 3. 违约用户和非违约用户在账单金额上的分布对比图（箱线图）
sns.boxplot(x='default', y='bill_amount', data=df, ax=axes[1, 0])
axes[1, 0].set_title('账单金额分布对比')
axes[1, 0].set_xticklabels(['不违约', '违约'])
axes[1, 0].set_xlabel('违约状态')
axes[1, 0].set_ylabel('账单金额')

# 隐藏空的子图
axes[1, 1].axis('off')

# 调整布局
plt.tight_layout()
plt.show()