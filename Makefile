HTMLDIR := docs/build/html

.PHONY: help
help:
	@echo "Please use \`make <target>\` where <target> is one of:"
	@echo "  docs         build 'latexpdf'+'html' documentation"
	@echo "  html         'apidoc', then build HTML doc webpages"
	@echo "  latexpdf     'apidoc', then build PDF docs"
	@echo "  clean        delete built documentation and other files"
	@echo "  apidoc       'clean' and rebuild automatic API"
	@echo "               documentation definitions"
	@echo "  ghup         push 'docs/build/html' to GitHub pages"
	@echo "  ghpages      make 'docs' and push to GitHub pages"

.PHONY: docs
docs: latexpdf html
	@echo "Generated all documentation."

.PHONY: html
html: apidoc
	cd docs && $(MAKE) html
	@echo "HTML builds located in docs/build."

.PHONY: latexpdf
latexpdf: apidoc
	cd docs && $(MAKE) latexpdf
	@echo "PDF builds located in docs/build."

.PHONY: clean
clean:
	[ -d docs/build ] && rm -vr docs/build || true
	find hpmoc bin -type d -name __pycache__ | xargs rm -fvr
	find hpmoc bin -type f -name '*.pyc' | xargs rm -fv
	rm -f docs/source/modules.rst
	find docs/source -name 'hpmoc*rst' | xargs rm -fv

.PHONY: apidoc
apidoc: clean
	$(eval ONLYHTML := '.. only:: html')
	sphinx-apidoc \
		--no-toc --module-first --implicit-namespaces --separate \
		--output-dir=docs/source hpmoc \
	# remove the submodule/subpackage headings to TOC depth
	sed -i.orig '/Subpackages/,/-*/s/^-*$$//' docs/source/hpmoc*.rst
	sed -i.orig '/Subpackages/s/^\(.*\)$$/.. only:: html'"$$(printf '\\\n\\\n\\\n')"'   **\1**/' docs/source/hpmoc*.rst
	sed -i.orig '/Submodules/,/-*/s/^-*$$//' docs/source/hpmoc*.rst
	sed -i.orig '/Submodules/s/^\(.*\)$$/.. only:: html'"$$(printf '\\\n\\\n\\\n')"'   **\1**/' docs/source/hpmoc*.rst
	printf '.. toctree::\n\n' >docs/source/hpmoc-subcomponents.rst
	find docs/source/ -not -name 'hpmoc.*.*.rst' -name 'hpmoc.*.rst' -exec basename {} \; | sort | sed -n 's/^/   /p' >>docs/source/hpmoc-subcomponents.rst
	python -c 'f="docs/source/hpmoc.rst"; h="\n\n".join(open(f).read().split("\n\n")[0:2]); open(f, "w").write(h)'
	rm docs/source/hpmoc*.rst.orig

.PHONY: ghup
ghup:
	$(eval TMP := docs/build/ghup)
	mkdir "$(TMP)"
	cp -R docs/build/html "$(TMP)"
	touch "$(TMP)"/html/.nojekyll
	echo hpmoc.stc.sh >"$(TMP)"/html/CNAME
	git -C "$(TMP)"/html/ init -b master
	git -C "$(TMP)"/html/ add .
	git -C "$(TMP)"/html/ commit -m 'docs'
	git -C "$(TMP)"/html/ remote add origin git@github.com:stefco/hpmoc.stc.sh.git
	git -C "$(TMP)"/html/ push -f -u origin master
	rm -rf "$(TMP)"

.PHONY: ghpages
ghpages: docs ghup
