"""Label the existing LF/HF field art as a conceptual refinement objective."""
from promo_drawing import Drawing, ASSETS, TEAL

def main():
    d=Drawing(360)
    d.text(28,24,'Schematic / Conceptual',18,bold=True)
    d.text(28,62,'Coarse field → refinement objective',26,bold=True)
    d.text(40,116,'Coarse LF field',20,bold=True)
    d.text(560,116,'Refined prediction (HF-like)',20,TEAL,bold=True)
    d.field(ASSETS/'lf-field-sample.png',(28,155,420,300))
    d.field(ASSETS/'hf-field-sample.png',(540,155,932,300))
    d.arrow((440,228),(520,228),TEAL)
    d.text(28,326,'Illustrative artwork · neither panel is a verified NNLCI checkpoint output',17)
    d.save('nnlci-lf-to-hf.png')

if __name__=='__main__': main()
