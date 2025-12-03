from img_caption import gen_img_cap
from attention_caption import get_attention_caption
from img_grad import get_img_grad
from style_trans import get_style_transfer

def main():
    content = "img/cat.jpg"
    style = "img/style.webp"
    att_cap = get_attention_caption(content)
    img_cap = gen_img_cap(content)
    get_img_grad(img=content, attention_caption=att_cap, img_caption=img_cap)
    get_style_transfer(content, style)

    pass



if __name__ =="__main__":
    main()
    print("Process Completed")