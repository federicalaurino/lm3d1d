.PHONY: all clean

all: mesh pdf

pdf: 3d1d_coupled_LM.tex
	latexmk -pdf -pdflatex="pdflatex --shell-escape %O %S" $<

mesh: conform_mesh.tex
	latexmk -pdf -pdflatex="pdflatex --shell-escape %O %S" $<

view: 
	evince 3d1d_coupled_LM.pdf &

clean:
	latexmk -C
