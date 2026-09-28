"""Redraw the legacy stencil/alignment concept with consistent readable type.

The preserved nnlci-local-stencil-base.png is the conceptual source. This
structured-grid redraw isolates its LF alignment mechanism, without the old
HF-to-point-selection arrow. It represents no numerical simulation.
"""
from promo_drawing import Drawing, BLUE, GREEN, INK

def main():
    d=Drawing(415)
    d.text(28,24,'Schematic / Conceptual',18,bold=True)
    d.text(28,58,'Align neighborhoods at one physical location',26,bold=True)
    for left,color,label,step in [(28,BLUE,'LF1 · coarse',42),(354,GREEN,'LF2 · finer',21)]:
        d.box((left,115,left+260,300),color)
        d.text(left+16,132,label,20,INK,bold=True)
        for x in range(left+25,left+239,step): d.line([(x,177),(x,277)])
        for y in range(180,278,step): d.line([(left+20,y),(left+240,y)])
        d.box((left+86,193,left+174,265),color, '#eef2f6')
        for x in [left+100,left+130,left+160]: d.dot(x,229,color)
        d.dot(left+130,229,INK,7)
    d.line([(158,302),(158,320),(638,320),(638,232)],INK,2)
    d.line([(484,302),(484,320)],INK,2)
    d.arrow((638,232),(650,232))
    d.text(656,145,'Align / interpolate',21,bold=True)
    d.text(656,188,'Collect local values',18)
    d.text(656,224,'Concatenate LF1 | LF2',18)
    d.text(656,264,'● Same target location',18)
    d.text(28,365,'Coordinates come from the target grid; inference point selection needs no HF truth.',17)
    d.save('nnlci-local-stencil-explained.png')

if __name__=='__main__': main()
