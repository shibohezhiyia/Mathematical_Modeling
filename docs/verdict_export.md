# 判决追溯与写作边界

`build_trace_bundle` 把判决、输入、IR、执行记录、证据和运行清单放入同一个可校验包，并显式报告缺失引用。`approved_writing_projection` 只接受有证据引用的显式 `approved` 候选；条件、未决和被反证结果不会进入写作 API。

该投影允许改写表达，不允许修改数字、假设、适用范围、反例或证据引用。

可选的 `certificate` 字段接受 `conclusion_certificate` 生成的有限证据证书，并校验版本；缺失证书不会被伪造为通过。
