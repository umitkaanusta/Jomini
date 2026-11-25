"""Dataset catalogue and citation registry, so every result can say where it came from.

>>> from jomini import catalog
>>> catalog.dataset("kursk").licence
'Public domain (U.S. Government work, 17 U.S.C. 105)'
>>> print(catalog.bibtex(["lucas2004", "bracken1995"]))

Each :class:`Dataset` records coverage, access (bundled or fetched at runtime),
licence, caveats and the references to cite. :data:`REFERENCES` also covers the
methods Jomini implements. Entries give DOIs and page ranges only where they were
checked; verify against the publisher before formal citation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from jomini import __version__

VERSION = __version__
"""Jomini version recorded in provenance blocks."""

Access = Literal["bundled", "fetched"]


@dataclass(frozen=True)
class Reference:
    key: str
    kind: Literal["article", "book", "techreport", "misc", "phdthesis", "mastersthesis"]
    fields: dict[str, str]

    def bibtex(self) -> str:
        body = ",\n".join(f"  {k} = {{{v}}}" for k, v in self.fields.items())
        return f"@{self.kind}{{{self.key},\n{body}\n}}"

    @property
    def short(self) -> str:
        """Author-year label, e.g. ``Lucas & Turkes (2004)``."""
        author = self.fields.get("author", self.fields.get("title", self.key))
        names = [a.split(",")[0].strip().strip("{}") for a in author.split(" and ")]
        who = (
            names[0]
            if len(names) == 1
            else (f"{names[0]} & {names[1]}" if len(names) == 2 else f"{names[0]} et al.")
        )
        return f"{who} ({self.fields.get('year', 'n.d.')})"


def _ref(key: str, kind: str, **fields: str) -> Reference:
    return Reference(key, kind, dict(fields))  # type: ignore[arg-type]


REFERENCES: dict[str, Reference] = {
    r.key: r
    for r in [
        # ---------------------------------------------------------------- theory
        _ref(
            "lanchester1916",
            "book",
            author="Lanchester, Frederick W.",
            title="Aircraft in Warfare: The Dawn of the Fourth Arm",
            publisher="Constable",
            address="London",
            year="1916",
        ),
        _ref(
            "bracken1995",
            "article",
            author="Bracken, Jerome",
            title="Lanchester Models of the Ardennes Campaign",
            journal="Naval Research Logistics",
            volume="42",
            number="4",
            pages="559--577",
            year="1995",
        ),
        _ref(
            "helmbold1964",
            "article",
            author="Helmbold, Robert L.",
            title="Some Observations on the Use of Lanchester's Theory for Prediction",
            journal="Operations Research",
            volume="12",
            number="5",
            pages="778--781",
            year="1964",
        ),
        _ref(
            "helmbold1971",
            "techreport",
            author="Helmbold, Robert L.",
            title="Decision in Battle: Breakpoint Hypotheses and Engagement Termination Data",
            institution="RAND Corporation",
            number="R-772-PR",
            year="1971",
        ),
        _ref(
            "hartley1995",
            "article",
            author="Hartley, Dean S., III",
            title="A Mathematical Model of Attrition Data",
            journal="Naval Research Logistics",
            volume="42",
            number="4",
            pages="585--607",
            year="1995",
        ),
        _ref(
            "fricker1998",
            "article",
            author="Fricker, Ronald D., Jr.",
            title="Attrition Models of the Ardennes Campaign",
            journal="Naval Research Logistics",
            volume="45",
            number="1",
            pages="1--22",
            year="1998",
        ),
        _ref(
            "lucas2004",
            "article",
            author="Lucas, Thomas W. and Turkes, T.",
            title="Fitting Lanchester Equations to the Battles of Kursk and Ardennes",
            journal="Naval Research Logistics",
            volume="51",
            number="1",
            pages="95--116",
            year="2004",
            url="https://hdl.handle.net/10945/44169",
        ),
        _ref(
            "hughes1995",
            "article",
            author="Hughes, Wayne P., Jr.",
            title="A Salvo Model of Warships in Missile Combat Used to Evaluate Their "
            "Staying Power",
            journal="Naval Research Logistics",
            volume="42",
            number="2",
            year="1995",
        ),
        _ref(
            "dupuy1979",
            "book",
            author="Dupuy, Trevor N.",
            title="Numbers, Predictions and War: Using History to Evaluate Combat Factors "
            "and Predict the Outcome of Battles",
            publisher="Bobbs-Merrill",
            address="Indianapolis",
            year="1979",
        ),
        _ref(
            "rotte2003",
            "article",
            author="Rotte, Ralph and Schmidt, Christoph M.",
            title="On the Production of Victory: Empirical Determinants of Battlefield "
            "Success in Modern War",
            journal="Defence and Peace Economics",
            volume="14",
            number="3",
            pages="175--192",
            year="2003",
        ),
        _ref(
            "brunton2016",
            "article",
            author="Brunton, Steven L. and Proctor, Joshua L. and Kutz, J. Nathan",
            title="Discovering Governing Equations from Data by Sparse Identification of "
            "Nonlinear Dynamical Systems",
            journal="Proceedings of the National Academy of Sciences",
            volume="113",
            number="15",
            pages="3932--3937",
            year="2016",
            doi="10.1073/pnas.1517384113",
        ),
        _ref(
            "rauch1965",
            "article",
            author="Rauch, H. E. and Tung, F. and Striebel, C. T.",
            title="Maximum Likelihood Estimates of Linear Dynamic Systems",
            journal="AIAA Journal",
            volume="3",
            number="8",
            pages="1445--1450",
            year="1965",
        ),
        _ref(
            "sindyukraine2024",
            "article",
            title="Analyzing Russia--Ukraine War Patterns Based on Lanchester Model Using "
            "SINDy Algorithm",
            journal="Mathematics",
            volume="12",
            number="6",
            pages="851",
            year="2024",
            doi="10.3390/math12060851",
        ),
        _ref(
            "hawkes1971",
            "article",
            author="Hawkes, Alan G.",
            title="Spectra of Some Self-Exciting and Mutually Exciting Point Processes",
            journal="Biometrika",
            volume="58",
            number="1",
            pages="83--90",
            year="1971",
        ),
        _ref(
            "lewis2012",
            "article",
            author="Lewis, Erik and Mohler, George and Brantingham, P. Jeffrey and "
            "Bertozzi, Andrea L.",
            title="Self-Exciting Point Process Models of Civilian Deaths in Iraq",
            journal="Security Journal",
            volume="25",
            number="3",
            pages="244--264",
            year="2012",
        ),
        _ref(
            "browning2024",
            "misc",
            author="Browning, Raiha and Patten, Hamish and Rousseau, Judith and Mengersen, Kerrie",
            title="Bayesian Spatiotemporal Modelling of Political Violence and Conflict "
            "Events Using Discrete-Time Hawkes Processes",
            howpublished="arXiv:2408.14940",
            year="2024",
        ),
        # ---------------------------------------------------------------- data
        _ref(
            "sundberg2013",
            "article",
            author="Sundberg, Ralph and Melander, Erik",
            title="Introducing the UCDP Georeferenced Event Dataset",
            journal="Journal of Peace Research",
            volume="50",
            number="4",
            pages="523--532",
            year="2013",
        ),
        _ref(
            "davies2026",
            "article",
            author="Davies, Shawn and Pettersson, Ther{\\'e}se and {\\\"O}berg, Magnus",
            title="Organized Violence 1989--2025, and Violent Political Protests",
            journal="Journal of Peace Research",
            year="2026",
            doi="10.1093/jopres/xjag046",
        ),
        _ref(
            "caa1991",
            "techreport",
            author="{U.S. Army Concepts Analysis Agency}",
            title="Database of Battles, Version 1990 (CDB90)",
            institution="U.S. Army Concepts Analysis Agency",
            number="AD1038724 (DTIC)",
            year="1991",
        ),
        _ref(
            "arnold_cdb90",
            "misc",
            author="Arnold, Jeffrey B.",
            title="CDB90 Battle Dataset (release 10.8.0)",
            howpublished="\\url{https://github.com/jrnold/CDB90}",
            doi="10.5281/zenodo.23928303",
            year="2026",
        ),
        _ref(
            "livermore1900",
            "book",
            author="Livermore, Thomas L.",
            title="Numbers and Losses in the Civil War in America, 1861--65",
            publisher="Houghton, Mifflin and Company",
            address="Boston",
            year="1900",
        ),
        _ref(
            "arnold_acw",
            "misc",
            author="Arnold, Jeffrey B.",
            title="American Civil War Battle Data",
            howpublished="\\url{https://github.com/jrnold/acw_battle_data}",
            doi="10.6084/m9.figshare.1515995",
        ),
        _ref(
            "ivaniuk_dataset",
            "misc",
            author="Ivaniuk, Petro",
            title="2022 Ukraine--Russia War Dataset",
            howpublished="\\url{https://github.com/PetroIvaniuk/2022-Ukraine-Russia-War-Dataset}",
            note="Daily Russian losses as reported by the General Staff of the Armed Forces "
            "of Ukraine",
        ),
        _ref(
            "oryx",
            "misc",
            author="Oryx",
            title="Attack on Europe: Documenting Equipment Losses During the Russian "
            "Invasion of Ukraine",
            howpublished="\\url{https://www.oryxspioenkop.com/}",
        ),
        _ref(
            "scarnecchia_oryx",
            "misc",
            author="Scarnecchia, Daniel",
            title="oryx\\_data",
            howpublished="\\url{https://github.com/scarnecchia/oryx_data}",
        ),
        _ref(
            "drake_oryx",
            "misc",
            author="Drake, Lee",
            title="Russia-Ukraine",
            howpublished="\\url{https://github.com/leedrake5/Russia-Ukraine}",
        ),
        _ref(
            "jomini",
            "misc",
            author='Usta, {\\"U}mit Kaan',
            title=f"Jomini: Lanchester Combat Models and Historical Battle Data "
            f"(version {__version__})",
            howpublished="\\url{https://github.com/umitkaanusta/Jomini}",
        ),
    ]
}


@dataclass(frozen=True)
class Dataset:
    id: str
    title: str
    coverage: str
    unit: str
    access: Access
    licence: str
    loader: str
    """Python call that loads it."""
    cite: tuple[str, ...]
    caveats: tuple[str, ...] = field(default=())

    def references(self) -> list[Reference]:
        return [REFERENCES[k] for k in self.cite]


DATASETS: dict[str, Dataset] = {
    d.id: d
    for d in [
        Dataset(
            "cdb90",
            "CDB90: CAA Database of Battles",
            "660 land battles, 1600-1982; strengths, casualties, duration, outcome, "
            "commanders, posture, analysts' advantage ratings, narratives (610)",
            "battle",
            "bundled",
            "CC0 1.0 (data); MIT (code)",
            "jomini.cdb90.battles()",
            ("caa1991", "arnold_cdb90"),
            (
                "Advantage ratings were assigned after the fact by analysts who knew outcomes.",
                "Two-thirds of battles are from 1900-1945; selection was for US Army studies.",
                "Duration is days of active combat, not calendar days.",
            ),
        ),
        Dataset(
            "kursk",
            "Battle of Kursk, southern front, daily",
            "5-18 July 1943, both sides: combat manpower, casualties, tanks, APCs, "
            "artillery on hand and lost",
            "day x side",
            "bundled",
            "Public domain (U.S. Government work, 17 U.S.C. 105)",
            "jomini.data.kursk()",
            ("lucas2004",),
            (
                "On-hand counts include reinforcements; losses include damaged equipment.",
                "14 days are too few to identify Lanchester exponents.",
            ),
        ),
        Dataset(
            "livermore",
            "American Civil War battles (Livermore 1900)",
            "63 major battles; strengths, killed+wounded, missing/captured, attacker, result",
            "battle",
            "bundled",
            "Public domain (1900); transcription ODC-BY 1.0",
            "jomini.data.livermore()",
            ("livermore1900", "arnold_acw"),
            (
                "Duration is calendar days, not active combat days.",
                "Battle selection differs from CDB90's Civil War battles.",
            ),
        ),
        Dataset(
            "ucdp_ged",
            "UCDP Georeferenced Event Dataset 26.1",
            "~418,000 events of organised violence worldwide, 1989-2025: date, place, "
            "conflict, dyad, deaths per side, civilian deaths, best/low/high estimates",
            "event",
            "fetched",
            "CC BY 4.0",
            "jomini.data.ucdp_ged()",
            ("sundberg2013", "davies2026"),
            (
                "Coded from news and other reports: coverage varies by place and time.",
                "Per-side deaths for Russia-Ukraine rely on sources that may echo each "
                "belligerent's claims; not independent of official figures.",
                "Events spanning several days are spread evenly over them when aggregated.",
            ),
        ),
        Dataset(
            "ukraine_general_staff",
            "Russian losses claimed by Ukraine's General Staff",
            "Daily since 25 Feb 2022; personnel, tanks, APCs, artillery, drones and more",
            "day (cumulative)",
            "fetched",
            "MIT (compilation)",
            "jomini.data.ukraine_general_staff()",
            ("ivaniuk_dataset",),
            ("One belligerent's unverified claims about the other.", "Russian losses only."),
        ),
        Dataset(
            "ukraine_oryx",
            "Oryx visually confirmed equipment losses (weekly)",
            "Feb 2022-Aug 2025 for Russia; Ukrainian side only from Mar 2025",
            "snapshot (cumulative)",
            "fetched",
            "CC BY 4.0 (compilation)",
            "jomini.data.ukraine_oryx(side)",
            ("oryx", "scarnecchia_oryx"),
            (
                "Counts only losses documented by photo or video: a lower bound.",
                "No longer updated after August 2025.",
            ),
        ),
        Dataset(
            "ukraine_oryx_events",
            "Oryx confirmed losses by event date (both sides)",
            "Daily, both sides, Feb 2022-Aug 2023",
            "day (events)",
            "fetched",
            "MIT (compilation)",
            "jomini.data.ukraine_oryx_events(side)",
            ("oryx", "drake_oryx"),
            (
                "Dates are when a loss was documented, which can lag the event.",
                "Documentation rates differ by side and over time.",
            ),
        ),
    ]
}

METHODS: dict[str, tuple[str, ...]] = {
    "lanchester": ("lanchester1916", "bracken1995"),
    "law_estimation": ("helmbold1964", "hartley1995"),
    "combat_power": ("dupuy1979", "rotte2003"),
    "breakpoints": ("helmbold1971",),
    "salvo": ("hughes1995",),
    "sindy": ("brunton2016", "sindyukraine2024"),
    "kalman": ("rauch1965",),
    "kursk_fits": ("lucas2004", "bracken1995", "fricker1998"),
    "hawkes": ("hawkes1971", "lewis2012", "browning2024"),
}


def dataset(key: str) -> Dataset:
    try:
        return DATASETS[key]
    except KeyError:
        raise KeyError(f"unknown dataset {key!r}; known: {', '.join(DATASETS)}") from None


def references_for(keys: Iterable[str]) -> list[Reference]:
    """References for any mix of reference keys, dataset ids and method names."""
    out: dict[str, Reference] = {}
    for key in keys:
        if key in REFERENCES:
            out[key] = REFERENCES[key]
        elif key in DATASETS:
            out.update({r.key: r for r in DATASETS[key].references()})
        elif key in METHODS:
            out.update({k: REFERENCES[k] for k in METHODS[key]})
        else:
            raise KeyError(f"unknown reference, dataset or method {key!r}")
    return list(out.values())


def bibtex(keys: Iterable[str], *, include_jomini: bool = True) -> str:
    """BibTeX for reference keys, dataset ids or method names (deduplicated)."""
    refs = references_for([*keys, *(["jomini"] if include_jomini else [])])
    return "\n\n".join(r.bibtex() for r in refs)
