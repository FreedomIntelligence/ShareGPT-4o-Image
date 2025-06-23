# ShareGPT-4o-Image: Aligning Multimodal Models with GPT-4o-Level Image Generation

<!--
<p align="center">
<img src="./assets/logo.jpg" alt="logo" width="170" class="center"/><br>
</p>
-->

<div align="center">
<h3>
  ShareGPT-4o-Image
</h3>
</div>

<p align="center">
📃 <a href="https://github.com/FreedomIntelligence/ShareGPT-4o-Image" target="_blank">Paper</a> | 📚 <a href="https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-4o-Image" target="_blank">ShareGPT-4o-Image</a> ｜🤗 <a href="https://huggingface.co/FreedomIntelligence/Janus-4o-7B" target="_blank">Janus-4o-7B</a>
</p>

## ⚡ Introduction

**ShareGPT-4o-Image** is a dataset comprising 45K *text-to-image* and 46K *text-and-image-to-image* pairs, all generated using GPT-4o’s advanced capabilities. This dataset enables the development of **Janus-4o**, a multimodal model that excels in both text-to-image and text-and-image-to-image generation, surpassing its predecessor, Janus-Pro. The release aims to propel open research in high-quality, instruction-guided image generation.

<div align=center>
<img src="./assets/fig_0.png"  width = "90%" alt="mainpic" align=center/>
</div>

## 🎨 Inference

### Gradio Demo

For the local gradio demo, you can run with the following command:

```
pip install -e .[gradio]

python demo/app_janus4o.py
```

## 👍 Acknowledgement
We greatly appreciate the following open-sourced repositories for providing us with valuable insights and tools: [Janus](https://github.com/deepseek-ai/Janus), [ImgEdit](https://github.com/PKU-YuanGroup/ImgEdit), [UltraEdit](https://github.com/HaozheZhao/UltraEdit), [ELLA](https://github.com/TencentQQGYLab/ELLA), [GenEval](https://github.com/djghosh13/geneval).
