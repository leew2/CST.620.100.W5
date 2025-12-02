from img_caption import gen_img_cap

def main():
    path = "img/cat.jpg"
    cap = gen_img_cap(path)
    print("BLIP Caption:", cap)



    pass



if __name__ =="__main__":
    main()
    print("Process Completed")