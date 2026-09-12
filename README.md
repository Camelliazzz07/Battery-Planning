# Battery Planning

所有脚本均可在 VS Code 中直接运行，且不依赖当前工作目录。

- 输入数据：`附件`
- 官方模板：`附件/附件5`
- 正式结果：`提交结果/result1.xlsx`、`result2.xlsx`、`result3.xlsx`
- 明细、图表和诊断：各问题目录下的`辅助输出`
- 可视化图片：统一为 SVG，便于论文缩放和排版

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

运行：

```powershell
python Code/问题一/solve_question1.py
python Code/问题二/q2_run_model.py
python Code/问题三/q3_run_model.py
```

可选检查：

```powershell
python Code/问题一/question1_preprocess_eda.py
python Code/问题二/q2_compare.py
python Code/问题三/test_q3_model.py
```
