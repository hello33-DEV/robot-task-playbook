# 资源准备与接口

## 密钥

包内只有空模板 `src/aic_competition/config/doubao.env.example`。需要联网识别时，复制成同目录 `doubao.env`，仅在本机填入 `DOUBAO_API_KEY`；也可以直接设置该环境变量。不要把终端中的密钥、请求头或真实env提交到GitHub。

其他兼容服务使用 `AIC_VISION_PROVIDER`、`AIC_VISION_BASE_URL`、`AIC_VISION_MODEL` 与 `AIC_VISION_API_KEY`；豆包使用 `DOUBAO_API_KEY`。客户端仍有原项目的provider约束，更换服务前应检查 `vision_config.py` 和请求格式。默认模型ID仅是历史配置，没有进行本次联网可用性或价格验证。

以往发给别人或进入过Git历史的密钥，应在服务商后台撤销/轮换；仅删除压缩包中的文件不会使旧密钥失效。

## 自己的模板图

本地匹配读取 `src/aic_competition/assets/cards/index.json`。每项包含 `file`、`math_answer`、`fruit_id`，例如：

```json
[{"file":"own_card.png","math_answer":2,"fruit_id":6}]
```

图片放在同目录；模板布局应符合当前算法的上文字、下图案掩膜。数学范围1—5、水果编码6—8是此项目的任务约定，不是通用视觉标签标准。若迁移到其他任务，必须同时调整解析、提示词、模板索引和播报映射。

没有随包分发官方PDF渲染图或照片。测试中临时生成的几何图案仅验证算法路径，不是训练集或真实题卡准确率。

## 自己录制的音频

把有使用权的PCM WAV放到 `src/aic_competition/assets/answer_audio/`：15种组合为 `answer_1_6.wav` 等，另外需要 `answer_test.wav` 和 `finish.wav`。

对应文案：组合音频“识别到旋转靶X，识别到移动靶Y。”；测试音频“语音测试，小车扬声器播放测试。这不是图片识别结果。”；结束音频“到达终点，比赛结束。”。录制后生成校验清单：

```bash
python tools/prepare_audio.py
```

此工具只检查已有文件并写manifest，不联网、不合成、不播放。没有有效录音时完整路线不能通过音频预检；不要用测试生成的静音文件充当真实播报。测试的静音夹具只存在临时目录。

## ROS外部服务契约

下列连接器的原文件带有禁止未经授权分发的声明，因此没有随包发布，也没有通过删除声明改成“自有代码”。可以使用合法取得的连接器，或按协议另行实现：

- API视觉：`/fruit_detection/reset`（std_srvs/Trigger）清空缓存；`/fruit_detection`（Trigger）轮询，空缓存为 `0,0`，成功消息为 `数学编号,水果编号`。`/detect=1`触发新识别，`/detect=255`结束；必须遵守 `/fruit_detection/capture_after` 的新帧时间限制。
- 本地视觉：`/competition/recognize`（Trigger）返回JSON，包含 `math_answer` 和 `fruit_id`；`local_vision.py` 已保留，但需要自备模板与cv_bridge。
- 播音：`/tts_service` 使用外部 `TTS_audio/StringService`，请求含 `text`、响应含 `result`。需要自备有权使用的服务定义和实现；失败、错误或空结果不能作为成功。此自定义服务包不在发布包内。
- 导航速度输入：`/competition/navigation_cmd_vel`；手动瞄准输入：`/competition/manual_cmd_vel`；安全桥输出到 `/cmd_vel`。底盘原节点不能绕过安全桥同时订阅另一套运动指令。

这些是现有适配器要求的契约，不代表相同名称的任意官方节点都满足。默认ROS启动仅做dry-run；真实驱动、地图、相机标定、目标检测和服务端需分别准备。
