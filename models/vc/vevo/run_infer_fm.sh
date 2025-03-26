save_root="/storage/zhangxueyao/workspace/Amphion/ckpts/Vevo"
model_name="amphionvevo_fm"

# 创建日志目录
log_dir="${save_root}/log_inference"
mkdir -p $log_dir

cd /storage/zhangxueyao/workspace/Amphion/

# # ===== Run (TTS) =====
# evalset_root="/storage/zhangxueyao/workspace/SpeechGenerationYC/EvalSet/tts"

# eval_setting="seedtts_en"
# task_name="tts"
# CUDA_VISIBLE_DEVICES=3 python -m models.vc.vevo.infer_vevotimbre_tts \
#     --save_root $save_root \
#     --model_name $model_name \
#     --evalset_root $evalset_root \
#     --eval_setting $eval_setting \
#     --task_name $task_name > "${log_dir}/${model_name}_${eval_setting}.log" 2>&1 &

# eval_setting="seedtts_zh"
# task_name="tts"
# CUDA_VISIBLE_DEVICES=4 python -m models.vc.vevo.infer_vevotimbre_tts \
#     --save_root $save_root \
#     --model_name $model_name \
#     --evalset_root $evalset_root \
#     --eval_setting $eval_setting \
#     --task_name $task_name > "${log_dir}/${model_name}_${eval_setting}.log" 2>&1 &

# ===== Run (VC) =====
evalset_root="/storage/zhangxueyao/workspace/SpeechGenerationYC/EvalSet/vc"

eval_setting="seedtts_vc_en"
task_name="vc"
CUDA_VISIBLE_DEVICES=3 python -m models.vc.vevo.infer_vevotimbre_tts \
    --save_root $save_root \
    --model_name $model_name \
    --evalset_root $evalset_root \
    --eval_setting $eval_setting \
    --task_name $task_name > "${log_dir}/${model_name}_${eval_setting}.log" 2>&1 &

eval_setting="seedtts_vc_zh"
task_name="vc"
CUDA_VISIBLE_DEVICES=4 python -m models.vc.vevo.infer_vevotimbre_tts \
    --save_root $save_root \
    --model_name $model_name \
    --evalset_root $evalset_root \
    --eval_setting $eval_setting \
    --task_name $task_name > "${log_dir}/${model_name}_${eval_setting}.log" 2>&1 &

# 等待所有后台进程完成
wait

echo "所有推理任务已完成，日志文件保存在 ${log_dir} 目录下"