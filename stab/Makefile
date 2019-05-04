.PHONY: all clean

all: pdf

pdf: stab.tex
	latexmk -pdf -pdflatex="pdflatex --shell-escape %O %S" $<

view: 
	evince stab.pdf &

clean:
	latexmk -C
