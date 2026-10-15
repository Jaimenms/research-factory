# Regras partilhadas de construção dos artigos (incluídas pelo Makefile de cada artigo em papers/<artigo>/).
#
# Alvos: pdf (omissão: figuras + compilação), figures, refigures, pdf-only, clean, distclean, check, open.
# Variáveis que o Makefile do artigo pode definir antes do include:
#   PAPER      nome do ficheiro .tex principal sem extensão (omissão: main)
#   BUILD      pasta de saída do LaTeX (omissão: build; fora do git)
#   FIGSCRIPT  guião que gera as figuras (omissão: figures/make_figures.py)
#   GEN        pasta das figuras geradas (omissão: figures/generated; fora do git)
#
# Motor: latexmk (TeX Live/MacTeX) se existir, senão tectonic (brew install tectonic; descarrega os
# pacotes à primeira compilação). Ambos correm o BibTeX e as passagens extra sozinhos.

PAPER     ?= main
BUILD     ?= build
FIGSCRIPT ?= figures/make_figures.py
GEN       ?= figures/generated

COMMON  := $(dir $(lastword $(MAKEFILE_LIST)))
PROJECT := $(abspath $(COMMON)/../..)
UV      ?= uv run --project $(PROJECT) python
TEXDIR  := $(abspath $(COMMON)/latex)   # ficheiros LaTeX partilhados (ícones da classe CAS)

LATEXMK  := $(shell command -v latexmk 2>/dev/null)
TECTONIC := $(shell command -v tectonic 2>/dev/null)

ifneq ($(LATEXMK),)
  ENGINE  := latexmk
  COMPILE  = TEXINPUTS="$(TEXDIR)//:" $(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -outdir=$(BUILD) $(PAPER).tex
else ifneq ($(TECTONIC),)
  ENGINE  := tectonic
  COMPILE  = $(TECTONIC) -Z search-path=$(TEXDIR) --outdir $(BUILD) --keep-logs --keep-intermediates $(PAPER).tex
else
  ENGINE  := none
  COMPILE  = $(error nenhum motor LaTeX encontrado: 'brew install tectonic' ou TeX Live com latexmk)
endif

.PHONY: all pdf pdf-only figures refigures clean distclean check open

all: pdf

# As figuras dependem do guião e do apoio comum; as corridas (runs/) não são seguidas pelo make:
# depois de novas corridas, `make refigures`.
figures: $(GEN)/.stamp

$(GEN)/.stamp: $(FIGSCRIPT) $(COMMON)figlib.py $(COMMON)cmfsim-paper.mplstyle
	$(UV) $(FIGSCRIPT) --out $(GEN)
	@touch $@

refigures:
	@rm -f $(GEN)/.stamp
	@$(MAKE) figures

pdf: figures
	@mkdir -p $(BUILD)
	$(COMPILE)
	@echo "PDF: $(CURDIR)/$(BUILD)/$(PAPER).pdf"

# Só o LaTeX (não toca nas figuras).
pdf-only:
	@mkdir -p $(BUILD)
	$(COMPILE)

clean:
	rm -rf $(BUILD)

distclean: clean
	rm -rf $(GEN)

check:
	@echo "motor LaTeX: $(ENGINE)"
	@echo "projeto:     $(PROJECT)"
	@echo "python:      $(UV)"
	@echo "figuras:     $(GEN)"

open: pdf
	open $(BUILD)/$(PAPER).pdf
