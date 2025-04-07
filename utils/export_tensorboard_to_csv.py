import argparse
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import os

def export_tensorboard_to_excel(in_path, out_path, keys=None):
    # 加载事件数据
    event_data = event_accumulator.EventAccumulator(in_path)
    event_data.Reload()
    
    # 如果没有指定keys，则使用所有可用的标量keys
    if keys is None:
        keys = event_data.Tags()['scalars']
    
    # 创建一个字典来存储所有数据
    data_dict = {}
    
    # 对于每个key，提取步骤和值
    for key in keys:
        events = event_data.Scalars(key)
        steps = [event.step for event in events]
        values = [event.value for event in events]
        
        # 将步骤和值添加到字典中
        if 'step' not in data_dict:
            data_dict['step'] = steps
        data_dict[key] = values
    
    # 创建DataFrame
    df = pd.DataFrame(data_dict)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # 导出到Excel文件
    df.to_excel(out_path, index=False)
    print(f"数据已成功导出到: {out_path}")

if __name__ == "__main__":
    # 创建三个空的DataFrame来存储不同指标的数据
    r1_df = pd.DataFrame()
    r5_df = pd.DataFrame()
    r10_df = pd.DataFrame()
    
    metrics = ['mapillary_dataset/R1', 'mapillary_dataset/R5', 'mapillary_dataset/R10']
    output_files = [
        'some_result_images/MQ_size/combined_R1_metrics.xlsx',
        'some_result_images/MQ_size/combined_R5_metrics.xlsx',
        'some_result_images/MQ_size/combined_R10_metrics.xlsx'
    ]
    
    for i in range(0, 15):
        parser = argparse.ArgumentParser(description='将TensorBoard日志导出为Excel文件')
        parser.add_argument('--in_path', type=str, default=f'logs/dinov2_backbone_dinov2_large/lightning_logs/version_{i}',
                            help='TensorBoard日志文件的路径')
        parser.add_argument('--out_path', type=str, default=f'some_result_images/MQ_size/version_{i}_mapillary_dataset_metrics.xlsx',
                            help='输出Excel文件的路径')
        parser.add_argument('--keys', nargs='+', default=metrics,
                            help='要导出的指标键列表')
        
        args = parser.parse_args()
        
        # 加载事件数据
        event_data = event_accumulator.EventAccumulator(args.in_path)
        event_data.Reload()
        
        # 提取每个指标的数据并添加到相应的DataFrame
        for idx, metric in enumerate(metrics):
            if metric in event_data.Tags()['scalars']:
                events = event_data.Scalars(metric)
                steps = [event.step for event in events]
                values = [event.value for event in events]
                
                temp_df = pd.DataFrame({'step': steps, f'version_{i}': values})
                
                # 根据指标类型添加到相应的DataFrame
                if idx == 0:  # R1
                    if r1_df.empty:
                        r1_df = temp_df
                    else:
                        r1_df = pd.merge(r1_df, temp_df, on='step', how='outer')
                elif idx == 1:  # R5
                    if r5_df.empty:
                        r5_df = temp_df
                    else:
                        r5_df = pd.merge(r5_df, temp_df, on='step', how='outer')
                elif idx == 2:  # R10
                    if r10_df.empty:
                        r10_df = temp_df
                    else:
                        r10_df = pd.merge(r10_df, temp_df, on='step', how='outer')
        
        # 同时也导出单个version的数据（保留原有功能）
        export_tensorboard_to_excel(args.in_path, args.out_path, args.keys)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_files[0]), exist_ok=True)
    
    # 导出合并后的数据到三个不同的Excel文件
    r1_df.to_excel(output_files[0], index=False)
    r5_df.to_excel(output_files[1], index=False)
    r10_df.to_excel(output_files[2], index=False)
    
    print(f"R1指标数据已合并导出到: {output_files[0]}")
    print(f"R5指标数据已合并导出到: {output_files[1]}")
    print(f"R10指标数据已合并导出到: {output_files[2]}")

