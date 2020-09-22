.PHONY: all clean

all: rev re

pdf: 3d1d_coupled_LM.tex
	latexmk -pdf -pdflatex="pdflatex --shell-escape %O %S" $<

rev: 3d1d_LM_rev.tex
	latexmk -pdf -pdflatex="pdflatex --shell-escape %O %S" $<

re: reply_to_reviewers.tex
	latexmk -pdf -pdflatex="pdflatex --shell-escape %O %S" $<

view: 
	evince 3d1d_LM_rev.pdf &
	evince reply_to_reviewers.pdf &

clean:
	latexmk -C
