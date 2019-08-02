.PHONY: all clean

all: pdf

pdf: solver.tex problem_omega.tex problem_gamma.tex coupled2d.tex coupled3d.tex
	latexmk -pdf -pdflatex="pdflatex --shell-escape %O %S" $<

view: 
	evince solver.pdf &

clean:
	latexmk -C
