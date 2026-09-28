"""Compose the refinement objective from existing illustrative field artwork."""
from promo_drawing import Drawing, ASSETS, TEAL

def main():
    d=Drawing(430)
    d.text(28,24,'Schematic / Conceptual',18,bold=True)
    d.text(28,59,'CFD-informed fidelity lift',30,bold=True)
    for box in [(28,120,306,365),(350,120,610,365),(654,120,932,365)]: d.box(box)
    d.text(46,140,'Coarse LF CFD',22,bold=True)
    d.field(ASSETS/'lf-field-sample.png',(40,185,294,335))
    d.text(370,140,'Learned refinement',20,bold=True)
    d.text(375,200,'LF1 + LF2',22,bold=True)
    d.arrow((480,238),(480,269),TEAL)
    d.text(375,286,'Local → field',21,TEAL,bold=True)
    d.text(674,137,'Refined prediction',20,TEAL,bold=True)
    d.text(674,164,'(HF-like)',18)
    d.field(ASSETS/'hf-field-sample.png',(666,198,920,342))
    d.arrow((309,245),(346,245)); d.arrow((614,245),(650,245),TEAL)
    d.text(28,392,'Illustrative field art · refinement objective, not checkpoint evidence',18)
    d.save('nnlci-concept-overview.png')

if __name__=='__main__': main()
