.PHONY: all clean

all: pdf

pdf: solver.tex problem_omega.tex problem_gamma.tex coupled.tex
	latexmk -pdf -pdflatex="pdflatex --shell-escape %O %S" $<

view: 
	evince solver.pdf &

clean:
	latexmk -C
