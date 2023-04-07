# line_seg = seg + (conf, ) * self.args.save_conf + (() if id is None else (id, ))
# with open(f'{self.txt_path + "_seg"}.txt', 'a') as f:
#     f.write(('%g ' * len(seg)).rstrip() % (seg) + '\n')
with open(f'{self.txt_path + "_seg"}.json', 'a') as f:
    json.dumps(seg.tolist())

bbox_xyxy = det.xyxy.clone().cpu().numpy().reshape(-1)
with open(f'{self.txt_path + "_bbox_xyxy"}.txt', 'a') as f:
    f.write(('%g ' * len(bbox_xyxy)).rstrip() % (bbox_xyxy) + '\n')

bbox_xywh = det.xywh.clone().cpu().numpy().reshape(-1)
with open(f'{self.txt_path + "_bbox_xywh"}.txt', 'a') as f:
    f.write(('%g ' * len(bbox_xywh)).rstrip() % (bbox_xywh) + '\n')
