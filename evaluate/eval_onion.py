import matplotlib.pyplot as plt

# 读取 record.log 文件
def read_log_file(file_path):
    bars = []
    attack_success_rates = []
    clean_accuracies = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            # 查找并提取 bar 值、attack success rate 和 clean acc
            if line.startswith('bar:'):
                bar = int(line.split(': ')[1].strip())
                bars.append(bar)
            elif line.startswith('attack success rate:'):
                attack_success_rate = float(line.split(': ')[1].strip())
                attack_success_rates.append(attack_success_rate)
            elif line.startswith('clean acc:'):
                clean_acc = float(line.split(': ')[1].strip())
                clean_accuracies.append(clean_acc)
    
    return bars, attack_success_rates, clean_accuracies

# 可视化并保存图片
def plot_and_save(bars, attack_success_rates, clean_accuracies, save_path):
    plt.figure(figsize=(10, 6))

    # 绘制 attack success rate 曲线
    plt.plot(bars, attack_success_rates, label='Attack Success Rate', color='red', marker='o', linestyle='-', linewidth=2)
    
    # 绘制 clean accuracy 曲线
    plt.plot(bars, clean_accuracies, label='Clean Accuracy', color='blue', marker='s', linestyle='--', linewidth=2)

    # 添加标题和标签
    plt.title('Attack Success Rate vs Clean Accuracy', fontsize=14)
    plt.xlabel('Bar Value', fontsize=12)
    plt.ylabel('Percentage', fontsize=12)
    
    # 显示网格
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # 添加图例
    plt.legend()

    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path)

    # 显示图像
    plt.show()

if __name__ == '__main__':
    # 设置文件路径和输出图像保存路径
    log_file_path = '/root/bdad/record.log'  # 你的日志文件路径
    output_image_path = 'attack_success_vs_clean_accuracy.png'  # 输出的图片文件路径
    
    # 读取 log 文件
    bars, attack_success_rates, clean_accuracies = read_log_file(log_file_path)
    
    # 绘制并保存图像
    plot_and_save(bars, attack_success_rates, clean_accuracies, output_image_path)
