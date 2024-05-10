from PIL import ImageTk, Image
import PIL
from tkinter import *    
import os
import numpy as np

WIDTH = 1280
HEIGHT = 800

class ExampleApp(Frame):
    def __init__(self,master, data_dir="./data"):
        Frame.__init__(self,master=None)
        self.data_dir = data_dir
        self.x = self.y = 0
        self.canvas = Canvas(master,  cursor="cross", width=WIDTH, height=HEIGHT)

        self.canvas.grid(row=0,column=0,sticky=N+S+E+W, rowspan=30)

        self.next_button = Button(root, text='Next', width=5, height=2, bd='10', command=self.next)
        self.next_button.grid(row=29,column=2,sticky=N+S+E+W)
        self.prev_button = Button(root, text='Prev', width=5, height=2, bd='10', command=self.prev)
        self.prev_button.grid(row=29,column=1,sticky=N+S+E+W)

        self.evaluate_button = Button(root, text='SAM', width=5, height=2, bd='10', command=self.sam)
        self.evaluate_button.grid(row=24,column=1,sticky=N+S+E+W, columnspan=2)
        self.evaluate_button = Button(root, text='SINet', width=5, height=2, bd='10', command=self.sinet)
        self.evaluate_button.grid(row=21,column=1,sticky=N+S+E+W, columnspan=2)
        self.evaluate_button = Button(root, text='SINet+Style', width=5, height=2, bd='10', command=self.sinet_style)
        self.evaluate_button.grid(row=22,column=1,sticky=N+S+E+W, columnspan=2)
        self.evaluate_button = Button(root, text='SINet+Synthetic', width=5, height=2, bd='10', command=self.sinet_synth)
        self.evaluate_button.grid(row=23,column=1,sticky=N+S+E+W, columnspan=2)
        
        self.evaluate_button = Button(root, text='Evaluate', width=5, height=2, bd='10', command=self.evaluate)
        self.evaluate_button.grid(row=28,column=1,sticky=N+S+E+W, columnspan=1)

        self.orig_button = Button(root, text='Original', width=5, height=2, bd='10', command=self.update_img)
        self.orig_button.grid(row=28,column=2,sticky=N+S+E+W, columnspan=1)

        self.hint1_button = Button(root, text='Show Hint 1', width=5, height=2, bd='10', command=self.hint1)
        self.hint1_button.grid(row=25,column=1,sticky=N+S+E+W, columnspan=1)
        
        self.hint2_button = Button(root, text='Show Hint 2', width=5, height=2, bd='10', command=self.hint2)
        self.hint2_button.grid(row=25,column=2,sticky=N+S+E+W, columnspan=1)

        self.display_res = Label(root, width=10, height=8)
        self.display_res.grid(row=26,column=1,sticky=N+S+E+W, columnspan=2, rowspan=2)

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.img_list = []
        self.img_pointer = 0
        # for img_name in os.listdir(os.path.join(self.data_dir, "images")):
        #     if ".DS" in img_name: continue
        #     self.img_list.append(img_name)

        self.img_list = [
            "COD10K-CAM-1-Aquatic-3-Crab-46.jpg",
            "animal-27.jpg",
            "camourflage_00018.jpg",
            "COD10K-CAM-2-Terrestrial-31-Gecko-1880.jpg",
            "COD10K-CAM-2-Terrestrial-31-Gecko-1913.jpg",
            "COD10K-CAM-2-Terrestrial-32-Giraffe-1932.jpg",
            "COD10K-CAM-2-Terrestrial-45-Spider-2576.jpg",
            "COD10K-CAM-2-Terrestrial-42-Sciuridae-2380.jpg",
            "COD10K-CAM-2-Terrestrial-45-Spider-2529.jpg",
            "COD10K-CAM-1-Aquatic-3-Crab-34.jpg",
            "COD10K-CAM-2-Terrestrial-30-Duck-1862.jpg",

            "COD10K-CAM-2-Terrestrial-26-Chameleon-1705.jpg",
            "COD10K-CAM-2-Terrestrial-34-Human-2010.jpg",
            "COD10K-CAM-2-Terrestrial-23-Cat-1367.jpg",
            "COD10K-CAM-2-Terrestrial-28-Deer-1788.jpg",
            "COD10K-CAM-2-Terrestrial-36-Leopard-2061.jpg",
            "COD10K-CAM-2-Terrestrial-23-Cat-1488.jpg",
            "COD10K-CAM-2-Terrestrial-29-Dog-1817.jpg",
            "COD10K-CAM-1-Aquatic-2-ClownFish-10.jpg",
            "COD10K-CAM-2-Terrestrial-31-Gecko-1923.jpg",
        ]

        self.rect = None
        self.gt_rect = None
        self.pred_rect = None

        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.bounding_box = None
 
        self.update_img()

    def update_img(self, image=None):
        self.canvas.delete('all')
        if image is None:
            self.im = PIL.Image.open(os.path.join(self.data_dir, "images", self.img_list[self.img_pointer]))
        else:
            self.im = image
        width, height = self.im.size
        old_width, old_height = self.im.size
        if width > WIDTH:
            height = height * (WIDTH/width)
            width = WIDTH
        if height > HEIGHT:
            width = width * (HEIGHT/height)
            height = HEIGHT
        width = int(width)
        height = int(height)
        self.im = self.im.resize((width, height))


        # self.canvas.config(scrollregion=(0,0,1000,1000))
        self.tk_im = ImageTk.PhotoImage(self.im)
        self.canvas.create_image(0,0,anchor="nw",image=self.tk_im) 

        if self.bounding_box is not None:
            self.rect = self.canvas.create_rectangle(self.bounding_box[0], self.bounding_box[1], self.bounding_box[2], self.bounding_box[3], fill="", outline="#0f0", width=4)

        self.scale_w = width/old_width
        self.scale_h = height/old_height


    def on_button_press(self, event):
        # save mouse drag start position
        self.start_x = event.x
        self.start_y = event.y

        if self.rect is not None:
            self.canvas.delete(self.rect)
            self.bounding_box = None
        # create rectangle if not yet exist
        self.rect = self.canvas.create_rectangle(self.x, self.y, 1, 1, fill="", outline="#0f0", width=4)

    def on_move_press(self, event):
        self.end_x, self.end_y = (event.x, event.y)

        # expand rectangle as you drag the mouse
        self.canvas.coords(self.rect, self.start_x, self.start_y, self.end_x, self.end_y)    


    def on_button_release(self, event):
        bounding_box = [
            self.start_x, self.start_y, self.end_x, self.end_y
        ]
        self.bounding_box = bounding_box
        pass  

    def next(self):
        self.bounding_box = None
        self.img_pointer = (self.img_pointer + 1) % len(self.img_list)
        self.update_img() 
        self.display_res.config(text = "")
    def prev(self):
        self.bounding_box = None
        self.img_pointer = (self.img_pointer - 1) % len(self.img_list)
        self.update_img() 
        self.display_res.config(text = "")

    def hint1(self):
        im = PIL.Image.open(os.path.join(self.data_dir, "images", self.img_list[self.img_pointer])).convert("RGB")
        gt = PIL.Image.open(os.path.join(self.data_dir, "gt", self.img_list[self.img_pointer].split(".")[0] + ".png"))
        gt = np.asarray(gt)
        xs = [x for x in range(gt.shape[1]) if gt[:, x].sum()]
        start_x = max(0, min(xs) - int(0.2 * gt.shape[1]))
        end_x = min(gt.shape[1], max(xs) + int(0.2 * gt.shape[1]))
        hint_mask = np.zeros(gt.shape)
        hint_mask[:, start_x:end_x] = 1
        masked_im = np.asarray(im) * hint_mask[..., None]
        masked_im = PIL.Image.fromarray(np.uint8(masked_im))
        # print(hint_mask.size, im.size)

        # masked_im = PIL.Image.composite(0, im, hint_mask)
        self.update_img(masked_im)

    def hint2(self):
        im = PIL.Image.open(os.path.join(self.data_dir, "images", self.img_list[self.img_pointer])).convert("RGB")
        gt = PIL.Image.open(os.path.join(self.data_dir, "gt", self.img_list[self.img_pointer].split(".")[0] + ".png"))
        gt = np.asarray(gt)
        xs = [x for x in range(gt.shape[1]) if gt[:, x].sum()]
        start_x = max(0, min(xs) - int(0.2 * gt.shape[1]))
        end_x = min(gt.shape[1], max(xs) + int(0.2 * gt.shape[1]))
        ys = [y for y in range(gt.shape[0]) if gt[y, :].sum()]
        start_y = max(0, min(ys) - int(0.2 * gt.shape[0]))
        end_y = min(gt.shape[0], max(ys) + int(0.2 * gt.shape[0]))
        hint_mask = np.zeros(gt.shape)
        hint_mask[start_y:end_y, start_x:end_x] = 1
        masked_im = np.asarray(im) * hint_mask[..., None]
        masked_im = PIL.Image.fromarray(np.uint8(masked_im))
        # print(hint_mask.size, im.size)

        # masked_im = PIL.Image.composite(0, im, hint_mask)
        self.update_img(masked_im)

    def sam(self):
        return self.show_prediction("sam")

    def sinet(self):
        return self.show_prediction("sinet_base")
    
    def sinet_style(self):
        return self.show_prediction("sinet_style")
    
    def sinet_synth(self):
        return self.show_prediction("sinet_synth")

    def show_prediction(self, dir_name):
        ext = ".jpg" if "sam" in dir_name.lower() else ".png"
        im = PIL.Image.open(os.path.join(self.data_dir, "images", self.img_list[self.img_pointer])).convert("RGB")
        gt = PIL.Image.open(os.path.join(self.data_dir, "gt", self.img_list[self.img_pointer].split(".")[0] + ".png"))
        pred = PIL.Image.open(os.path.join(self.data_dir, dir_name, self.img_list[self.img_pointer].split(".")[0] + ext))
        masked_im = PIL.Image.composite(0, im, PIL.ImageOps.invert(pred))
        self.update_img(masked_im)

        gt = np.asarray(gt) / 255
        gt_ys = [y for y in range(gt.shape[0]) if gt[y, :].sum()]
        gt_xs = [x for x in range(gt.shape[1]) if gt[:, x].sum()]
        gt_x1, gt_x2 = min(gt_xs), max(gt_xs)
        gt_y1, gt_y2 = min(gt_ys), max(gt_ys)

        pred = np.asarray(pred) / 255
        pred = pred > 0.5
        pred_ys = [y for y in range(pred.shape[0]) if pred[y, :].sum()]
        pred_xs = [x for x in range(pred.shape[1]) if pred[:, x].sum()]
        pred_x1, pred_x2 = min(pred_xs), max(pred_xs)
        pred_y1, pred_y2 = min(pred_ys), max(pred_ys)

        x_left = max(gt_x1, pred_x1)
        y_top = max(gt_y1, pred_y1)
        x_right = min(gt_x2, pred_x2)
        y_bottom = min(gt_y2, pred_y2)

        if x_right < x_left or y_bottom < y_top:
            intersection_area = 0
        else:
            intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
        bb2_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

        seg_iou = (pred * gt).sum()/((pred + gt)>0).sum()
            
        # self.display_res.insert(END, "Box-IOU: {:.2f}%".format(100*iou))
        self.display_res.config(text = "Box-IOU: {:.2f}%\nSeg-IOU: {:.2f}%".format(100*iou, 100*seg_iou))

        if self.gt_rect is not None:
            self.canvas.delete(self.gt_rect)
        if self.pred_rect is not None:
            self.canvas.delete(self.pred_rect)
        self.gt_rect = self.canvas.create_rectangle(self.scale_w*gt_x1, self.scale_h*gt_y1, self.scale_w*gt_x2, self.scale_h*gt_y2, fill="", outline="#00f", width=4)
        self.pred_rect = self.canvas.create_rectangle(self.scale_w*pred_x1, self.scale_h*pred_y1, self.scale_w*pred_x2, self.scale_h*pred_y2, fill="", outline="#f0f", width=4)



    def evaluate(self):
        im = PIL.Image.open(os.path.join(self.data_dir, "images", self.img_list[self.img_pointer]))
        gt = PIL.Image.open(os.path.join(self.data_dir, "gt", self.img_list[self.img_pointer].split(".")[0] + ".png"))
        masked_im = PIL.Image.composite(0, im, PIL.ImageOps.invert(gt))
        # im = np.asarray(im)
        # gt = np.asarray(gt)[..., None]
        # masked_im = im * gt #+ 0.5 * im * (1-gt)
        # masked_im = cv2.
        # masked_im = PIL.Image.fromarray(np.uint8(masked_im))
        self.update_img(masked_im)

        gt = np.asarray(gt)
        ys = [y for y in range(gt.shape[0]) if gt[y, :].sum()]
        xs = [x for x in range(gt.shape[1]) if gt[:, x].sum()]

        if self.gt_rect is not None:
            self.canvas.delete(self.gt_rect)

        self.gt_rect = self.canvas.create_rectangle(self.scale_w*min(xs), self.scale_h*min(ys), self.scale_w*max(xs), self.scale_h*max(ys), fill="", outline="#00f", width=4)
        if self.bounding_box is not None:
            self.rect = self.canvas.create_rectangle(self.bounding_box[0], self.bounding_box[1], self.bounding_box[2], self.bounding_box[3], fill="", outline="#0f0", width=4)

            gt_x1, gt_x2 = min(xs), max(xs)
            gt_y1, gt_y2 = min(ys), max(ys)

            pred_x1, pred_x2 = min(self.bounding_box[::2]), max(self.bounding_box[::2])
            pred_y1, pred_y2 = min(self.bounding_box[1::2]), max(self.bounding_box[1::2])

            pred_x1 = pred_x1 / self.scale_w
            pred_x2 = pred_x2 / self.scale_w
            pred_y1 = pred_y1 / self.scale_h
            pred_y2 = pred_y2 / self.scale_h


            x_left = max(gt_x1, pred_x1)
            y_top = max(pred_y1, pred_y1)
            x_right = min(gt_x2, pred_x2)
            y_bottom = min(gt_y2, pred_y2)

            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            if x_right < x_left or y_bottom < y_top:
                intersection_area = 0
            else:
                intersection_area = (x_right - x_left) * (y_bottom - y_top)

            # compute the area of both AABBs
            bb1_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
            bb2_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
            
            iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
                
            # self.display_res.insert(END, "Box-IOU: {:.2f}%".format(100*iou))
            self.display_res.config(text = "Box-IOU: {:.2f}%".format(100*iou))


if __name__ == "__main__":
    root=Tk()
    app = ExampleApp(root)
    root.mainloop()