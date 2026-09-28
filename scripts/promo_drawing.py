"""Shared 2x Pillow drawing primitives; coordinates are display pixels at 960px."""
from pathlib import Path
import math
from PIL import Image, ImageDraw, ImageFont
ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'docs/assets'
INK, MUTED, GRID = '#172033', '#465366', '#cbd5e1'
BLUE, GREEN, RED, TEAL = '#2563EB', '#159947', '#DC2626', '#087f8c'

class Drawing:
    def __init__(self, height):
        self.im = Image.new('RGB', (1920, height*2), 'white')
        self.d = ImageDraw.Draw(self.im)
    def text(self, x, y, label, size=18, color=INK, bold=False):
        name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
        font = ImageFont.truetype(name, size*2)
        self.d.text((x*2,y*2), label, font=font, fill=color, anchor='lt', spacing=10)
    def box(self, box, color=GRID, fill='#f8fafc'):
        self.d.rounded_rectangle(tuple(v*2 for v in box), radius=18, fill=fill, outline=color, width=4)
    def line(self, points, color=GRID, width=1):
        self.d.line([(x*2,y*2) for x,y in points], fill=color, width=width*2)
    def arrow(self, start, end, color=INK):
        self.line([start,end],color,2)
        a=math.atan2(end[1]-start[1],end[0]-start[0]); x,y=end
        self.d.polygon([(x*2,y*2),((x-10*math.cos(a)+5*math.sin(a))*2,(y-10*math.sin(a)-5*math.cos(a))*2),((x-10*math.cos(a)-5*math.sin(a))*2,(y-10*math.sin(a)+5*math.cos(a))*2)],fill=color)
    def dot(self,x,y,color=INK,r=5):
        self.d.ellipse(((x-r)*2,(y-r)*2,(x+r)*2,(y+r)*2),fill=color)
    def field(self,path,box):
        from PIL import ImageOps
        with Image.open(path) as source:
            image=ImageOps.contain(source.convert('RGB'),((box[2]-box[0])*2,(box[3]-box[1])*2))
            self.im.paste(image,(box[0]*2+(box[2]-box[0])*1-image.width//2,box[1]*2+(box[3]-box[1])*1-image.height//2))
    def save(self,name):
        self.im.save(ASSETS/name, optimize=True)
