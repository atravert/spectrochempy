# -*- coding: utf-8 -*-

#  =====================================================================================================================
#  Copyright (©) 2015-$today.year LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory
#  =====================================================================================================================
#

"""
Clean, build, and release the HTML and PDF documentation for SpectroChemPy.
```bash
  python make.py [options]
```
where optional parameters indicates which job(s) is(are) to perform.
"""
from os import environ

import argparse
import shutil
import sys
import warnings
import zipfile
from pathlib import Path
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
from sphinx.application import Sphinx
from sphinx.deprecation import RemovedInSphinx50Warning, RemovedInSphinx40Warning

from spectrochempy import version
from spectrochempy.utils import sh

warnings.filterwarnings(action='ignore', module='matplotlib', category=UserWarning)
warnings.filterwarnings(action='ignore', category=RemovedInSphinx50Warning)
warnings.filterwarnings(action='ignore', category=RemovedInSphinx40Warning)

# CONSTANT
PROJECTNAME = "spectrochempy"
REPO_URI = f"spectrochempy/{PROJECTNAME}"
API_GITHUB_URL = "https://api.github.com"
URL_SCPY = "www.spectrochempy.fr"

# GENERAL PATHS
DOCS = Path(__file__).parent
TEMPLATES = DOCS / '_templates'
PROJECT = DOCS.parent
DOCREPO = Path().home() / "spectrochempy_docs"
DOCTREES = DOCREPO / "~doctrees"
HTML = DOCREPO / "html"
LATEX = DOCREPO / 'latex'
DOWNLOADS = HTML / 'downloads'
SOURCES = PROJECT / PROJECTNAME

# DOCUMENTATION SRC PATH
SRC = DOCS
USERGUIDE = SRC / 'userguide'
GETTINGSTARTED = SRC / 'gettingstarted'
DEVGUIDE = SRC / 'devguide'
REFERENCE = SRC / 'userguide' / 'reference'

# generated by sphinx
API = REFERENCE / 'generated'
GALLERY = GETTINGSTARTED / 'gallery'

PY = list(SRC.glob(('**/*.py')))
for f in PY[:]:
    if (
            'generated' in f.parts or '.ipynb_checkpoints' in f.parts or 'gallery' in f.parts or 'examples' in
            f.parts or 'sphinxext' in f.parts):
        PY.remove(f)
    if f.name in ['conf.py', 'make.py']:
        PY.remove(f)

__all__ = []


# %%
class BuildDocumentation(object):

    # ..................................................................................................................
    def __init__(self):
        # determine if we are in the developement branch (latest) or master (stable)

        if 'dev' in version:
            self._doc_version = 'latest'
        else:
            self._doc_version = 'stable'

    # ..................................................................................................................
    @property
    def doc_version(self):
        return self._doc_version

    # ..................................................................................................................
    def __call__(self):

        parser = argparse.ArgumentParser()

        parser.add_argument("-H", "--html", help="create html pages", action="store_true")
        parser.add_argument("-P", "--pdf", help="create pdf manual", action="store_true")
        parser.add_argument("--clean", help="clean for a full regeneration of the documentation", action="store_true")
        parser.add_argument("--delnb", help="delete all ipynb", action="store_true")
        parser.add_argument("-m", "--message", default='DOCS: updated', help='optional git commit message')
        parser.add_argument("--api", help="execute a full regeneration of the api", action="store_true")
        parser.add_argument("-R", "--release", help="release the current version documentation on website",
                            action="store_true")
        parser.add_argument("--all", help="Build all docs", action="store_true")

        args = parser.parse_args()

        if len(sys.argv) == 1:
            parser.print_help(sys.stderr)
            return

        self.regenerate_api = args.api

        if args.clean and args.html:
            self.clean('html')

        if args.clean and args.pdf:
            self.clean('latex')

        if args.delnb:
            self.delnb()

        if args.html:
            self.make_docs('html')
            self.make_tutorials()

        if args.pdf:
            self.make_docs('latex')
            self.make_pdf()

        if args.all:
            self.delnb()
            self.make_docs('html', clean=True)
            self.make_docs('latex', clean=True)
            self.make_pdf()
            self.make_tutorials()

    @staticmethod
    def _confirm(action):
        # private method to ask user to enter Y or N (case-insensitive).
        answer = ""
        while answer not in ["y", "n"]:
            answer = input(f"OK to continue `{action}` Y[es]/[N[o] ? ", ).lower()
        return answer[:1] == "y"

    # ..................................................................................................................
    def make_docs(self, builder='html', clean=False):
        # Make the html or latex documentation

        doc_version = self.doc_version

        self.make_changelog()

        if builder not in ['html', 'latex']:
            raise ValueError('Not a supported builder: Must be "html" or "latex"')

        BUILDDIR = DOCREPO / builder
        print(f'{"-" * 80}\n'
              f'building {builder.upper()} documentation ({doc_version.capitalize()} version : {version})'
              f'\n{"-" * 80}')

        # recreate dir if needed
        if clean:
            print('CLEAN:')
            self.clean(builder)
        self.make_dirs()

        # update modified notebooks
        self.sync_notebooks()

        shutil.rmtree(API, ignore_errors=True)
        print(f'remove {API}')

        # run sphinx
        print(f'\n{builder.upper()} BUILDING:')
        srcdir = confdir = DOCS
        outdir = f"{BUILDDIR}/{doc_version}"
        doctreesdir = f"{DOCTREES}/{doc_version}"
        sp = Sphinx(srcdir, confdir, outdir, doctreesdir, builder)
        sp.verbosity = 1
        sp.build()

        print(f"\n{'-' * 130}\nBuild finished. The {builder.upper()} pages "
              f"are in {outdir}.")

        if doc_version == 'stable':
            doc_version = 'latest'
            # make also the lastest identical
            print(f'\n{builder.upper()} BUILDING:')
            srcdir = confdir = DOCS
            outdir = f"{BUILDDIR}/{doc_version}"
            doctreesdir = f"{DOCTREES}/{doc_version}"
            sp = Sphinx(srcdir, confdir, outdir, doctreesdir, builder)
            sp.verbosity = 1
            sp.build()

            print(f"\n{'-' * 130}\nBuild 'latest' finished. The {builder.upper()} pages "
                  f"are in {outdir}.")
            doc_version = 'stable'

        if builder == 'html':
            self.make_redirection_page()

        # a workaround to reduce the size of the image in the pdf document
        # TODO: v.0.2 probably better solution exists?
        if builder == 'latex':
            self.resize_img(GALLERY, size=580.)

    # ..................................................................................................................
    @staticmethod
    def resize_img(folder, size):
        # image resizing mainly for pdf doc

        for filename in folder.rglob('**/*.png'):
            image = imread(filename)
            h, l, c = image.shape
            ratio = 1.
            if l > size:
                ratio = size / l
            if ratio < 1:
                # reduce size
                image_resized = resize(image, (int(image.shape[0] * ratio), int(image.shape[1] * ratio)),
                                       anti_aliasing=True)
                imsave(filename, (image_resized * 255.).astype(np.uint8))

    # ..................................................................................................................
    def make_pdf(self):
        # Generate the PDF documentation

        doc_version = self.doc_version
        LATEXDIR = LATEX / doc_version
        print('Started to build pdf from latex using make.... '
              'Wait until a new message appear (it is a long! compilation) ')

        print('FIRST COMPILATION:')
        sh(f"cd {LATEXDIR}; lualatex -synctex=1 -interaction=nonstopmode spectrochempy.tex")

        print('MAKEINDEX:')
        sh(f"cd {LATEXDIR}; makeindex spectrochempy.idx")

        print('SECOND COMPILATION:')
        sh(f"cd {LATEXDIR}; lualatex -synctex=1 -interaction=nonstopmode spectrochempy.tex")

        print("move pdf file in the download dir")
        sh(f"cp {LATEXDIR / PROJECTNAME}.pdf {DOWNLOADS}/{doc_version}-{PROJECTNAME}.pdf")

    # ..................................................................................................................
    def sync_notebooks(self, pyfiles=PY):
        # Use  jupytext to sync py and ipynb files in userguide and tutorials
        print(f'\n{"-" * 80}\nSync *.py and *.ipynb using jupytex\n{"-" * 80}')
        count = 0
        for item in pyfiles:
            difftime = 1
            print(f'sync: {item.name}')
            if item.with_suffix('.ipynb').exists():
                difftime = item.stat().st_mtime - item.with_suffix('.ipynb').stat().st_mtime
            if difftime > .5:
                # may be modified
                count += 1
                sh.jupytext("--update-metadata", '{"jupytext": {"notebook_metadata_filter":"all"}}', "--set-formats",
                            "ipynb,py:percent", "--sync", item, silent=False)
            else:
                print('\tNo sync needed.')
        if count == 0:
            print('\nAll notebooks are up-to-date and synchronised with py files')
        print('\n')

    # ..................................................................................................................
    def delnb(self):
        # Remove all ipynb before git commit

        for nb in SRC.rglob('**/*.ipynb'):
            sh.rm(nb)
        for nbch in SRC.glob('**/.ipynb_checkpoints'):
            sh(f'rm -r {nbch}')

    # ..................................................................................................................
    def make_tutorials(self):
        # make tutorials.zip

        # clean notebooks output
        for nb in DOCS.rglob('**/*.ipynb'):
            # This will erase all notebook output
            sh(f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {nb}", silent=True)

        # make zip of all ipynb
        def zipdir(path, dest, ziph):
            # ziph is zipfile handle
            for nb in path.rglob('**/*.ipynb'):
                if '.ipynb_checkpoints' in nb.parent.suffix:
                    continue
                basename = nb.stem
                sh(f"jupyter nbconvert {nb} --to notebook"
                   f" --ClearOutputPreprocessor.enabled=True"
                   f" --stdout > out_{basename}.ipynb")
                sh(f"rm {nb}", silent=True)
                sh(f"mv out_{basename}.ipynb {nb}", silent=True)
                arcnb = str(nb).replace(str(path), str(dest))
                ziph.write(nb, arcname=arcnb)

        zipf = zipfile.ZipFile('~notebooks.zip', 'w', zipfile.ZIP_STORED)
        zipdir(SRC, 'notebooks', zipf)
        zipdir(GALLERY / 'auto_examples', Path('notebooks') / 'examples', zipf)
        zipf.close()

        sh(f"mv ~notebooks.zip {DOWNLOADS}/{self.doc_version}-{PROJECTNAME}-notebooks.zip")

    # ..................................................................................................................
    def make_redirection_page(self, ):
        # create an index page a the site root to redirect to latest version

        html = f"""
        <html>
        <head>
        <title>Redirect to the dev version of the documentation</title>
        <meta http-equiv="refresh" content="0; URL=https://{URL_SCPY}/latest">
        </head>
        <body>
        <p>
        We have moved away from the <strong>spectrochempy.github.io</strong> domain.
        If you're not automatically redirected, please visit us at
        <a href="https://{URL_SCPY}">https://{URL_SCPY}</a>.
        </p>
        </body>
        </html>
        """
        with open(HTML / 'index.html', 'w') as f:
            f.write(html)

    # ..................................................................................................................
    def clean(self, builder):
        # Clean/remove the built documentation.

        print(f'\n{"-" * 80}\nCleaning\n{"-" * 80}')

        doc_version = self.doc_version

        if builder == 'html':
            shutil.rmtree(HTML / doc_version, ignore_errors=True)
            print(f'remove {HTML / doc_version}')
            shutil.rmtree(DOCTREES / doc_version, ignore_errors=True)
            print(f'remove {DOCTREES / doc_version}')
            shutil.rmtree(API, ignore_errors=True)
            print(f'remove {API}')
            shutil.rmtree(GALLERY, ignore_errors=True)
            print(f'remove {GALLERY}')
        else:
            shutil.rmtree(LATEX / doc_version, ignore_errors=True)
            print(f'remove {LATEX / doc_version}')

    # ..................................................................................................................
    def make_dirs(self):
        # Create the directories required to build the documentation.

        doc_version = self.doc_version

        # Create regular directories.
        build_dirs = [DOCREPO, DOCTREES, HTML, LATEX, DOCTREES / doc_version, HTML / doc_version, LATEX / doc_version,
                      DOWNLOADS, ]
        for d in build_dirs:
            if not d.exists():
                print(f'Make dir {d}')
                Path.mkdir(d, exist_ok=False)

    # ..................................................................................................................
    def make_changelog(self):

        print(f'\n{"-" * 80}\nMake `changelogs`\n{"-" * 80}')

        outfile = REFERENCE / 'changelog.rst'

        sh.pandoc(PROJECT / 'CHANGELOG.md', '-f', 'markdown', '-t', 'rst', '-o', outfile)

        print(f'`Complete what\'s new` log written to:\n{outfile}\n')


# %%
Build = BuildDocumentation()

# %%
if __name__ == '__main__':

    environ['DOC_BUILDING'] = 'yes'
    Build()
    del environ['DOC_BUILDING']
