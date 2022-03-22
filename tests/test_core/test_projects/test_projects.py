# -*- coding: utf-8 -*-
# flake8: noqa

from spectrochempy.core.project.project import Project
from spectrochempy.core.scripts.script import Script, run_script
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core import preferences, INPLACE

prefs = preferences


# Basic
# --------------------------------------------------------------------------------------------------------


def test_project(ds1, ds2, dsm):
    myp = Project(name="AGIR processing")

    ds1.name = "toto"
    ds2.name = "tata"
    dsm.name = "titi"

    ds = ds1[:, 10, INPLACE]
    assert ds1.shape == ds.shape
    assert ds is ds1

    myp.add(ds1, ds2, dsm)

    assert myp.names[-1] == "titi"
    # AT: no parent for the moment
    # assert ds1.parent == myp

    # iteration
    d = []
    for item in myp:
        d.append(item)

    assert d[1] == "tata"

    ##
    # add sub project
    msp1 = Project(name="AGIR ATG")
    msp1.add(ds1)
    # AT: no parenty for the moment
    # assert ds1.parent == msp1  # ds1 has changed of project

    # A.T.: I don't see the motivation for such a behaviour:
    # assert ds1.name not in myp.datasets_names

    msp2 = Project(name="AGIR IR")

    myp.add(msp1, msp2)

    # an object can be accessed by it's name whatever it's type
    assert "tata" in myp.names
    assert myp["titi"] == dsm

    # import multiple objects in Project
    myp2 = Project(msp1, msp2, ds1, ds2)  # multi dataset and project and no names
    assert myp2.names == [msp1.name, msp2.name, ds1.name, ds2.name]


def test_empty_project():
    proj = Project(name="XXX")
    assert proj.name == "XXX"
    assert str(proj).strip() == "XXX (empty Project)"


def test_project_with_script():
    # Example from tutorial agir notebook
    # AT: does not deal with meta for now
    # proj = Project(
    #     Project(name="P350", label=r"$\mathrm{M_P}\,(623\,K)$"),
    #     Project(name="A350", label=r"$\mathrm{M_A}\,(623\,K)$"),
    #     Project(name="B350", label=r"$\mathrm{M_B}\,(623\,K)$"),
    #     name="HIZECOKE_TEST",
    # )
    proj = Project(
        Project(name="P350"),
        Project(name="A350"),
        Project(name="B350"),
        name="HIZECOKE_TEST",
    )

    assert proj.names == ["P350", "A350", "B350"]

    # add a dataset to a subproject
    ir = NDDataset([1, 2, 3])
    tg = NDDataset([1, 3, 4])

    # AT: to do
    proj.A350["IR"] = ir
    proj["TG"] = tg

    print(proj.A350)
    print(proj)
    # print(proj.A350.label)

    f = proj.save(confirm=False)

    newproj = Project.load("HIZECOKE_TEST")
    # print(newproj)
    assert str(newproj) == str(proj)
    assert newproj.A350.label == proj.A350.label

    # proj = Project.load('HIZECOKE')
    # assert proj.projects_names == ['A350', 'B350', 'P350']

    script_source = (
        "set_loglevel(INFO)\n"
        'info_("samples contained in the project are : '
        '%s"%proj.projects_names)'
    )

    proj["print_info"] = Script("print_info", script_source)

    # print(proj)

    # save but do not change the original data
    proj.save_as("HIZECOKE_TEST", overwrite_data=False)

    newproj = Project.load("HIZECOKE_TEST")

    # execute
    run_script(newproj.print_info, locals())
    newproj.print_info.execute(locals())

    newproj.print_info(locals())

    # attempts to resolve locals
    newproj.print_info()

    proj.save_as("HIZECOKE_TEST")
    newproj = Project.load("HIZECOKE_TEST")


def test_save_and_load_project(ds1, ds2):
    myp = Project(name="process")

    ds1.name = "toto"
    ds2.name = "tata"

    myp.add(ds1, ds2)

    fn = myp.save(confirm=False)

    proj = Project.load(fn)

    assert str(proj["toto"]) == "NDDataset: [float64] a.u. (shape: (z:10, y:100, x:3))"
