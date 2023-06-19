## All In One: General Multimodal Large Language Model

This reporsitory introduces a new visual information incorporation strategy, referred to as "Recall" mechanism for multimoldal language model. What's more, the model additionly supports all CV tasks including detection, segementation and more with additional special tokens. 

<p align="center">
     <img src="figures/flow.png" alt="AIO framework" width = "400">
     <br/>
     <sub><em>
     Overview of the proposed AIO framework.
    </em></sub>
</p>

As shown in the Figure, global visual information is first concatenated with textual embedding to provide a coarse cues. When generate the token, low-level patches are recalled and contexted to generate the next token. 





