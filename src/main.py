from img_caption import gen_img_cap
from attention_caption import get_attention_caption

def main():
    path = "img/cat.jpg"
    cap = get_attention_caption(path)
    print("Attention Caption:", cap)



    pass



if __name__ =="__main__":
    main()
    print("Process Completed")