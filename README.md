[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# DeepChat
Chatbot framework using [Huggingface](https://github.com/huggingface) transformers

This is meant to be a wrapper around huggingface, or potentially other frameworks and models to make chatbot development easier and fun to play around with.

# Supported Models
- [DialoGPT](https://huggingface.co/microsoft/DialoGPT-medium)

## Usage
TBD

## Examples
How to start a simple chatbot in a few lines of code using one of the pretrained Huggingface models. 

```python
from deepchat import DeepChat

# create DeepChat object that uses small model and default settings
dc = DeepChat("dialogpt", device_type="cuda")

# begins the chatbot conversation
dc.run()
```
