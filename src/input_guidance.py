"""Readable input guidance and conservative, non-causal device checks."""
from __future__ import annotations

import re
from .feature_engineering import ABSENT, MEASURE_FIELDS, is_missing, measurement_summary, tokens

FIELD_GUIDE = {
    'Cell_stack_sequence': ('Full device stack', 'Layers from substrate to top contact, separated by |. Example: SLG | FTO | TiO2 | Perovskite | Spiro-MeOTAD | Au.'),
    'Cell_architecture': ('Device architecture', 'nip: electron transport layer before the absorber; pin: hole transport layer before the absorber, in substrate-to-contact order.'),
    'Substrate_stack_sequence': ('Substrate layers', 'Recorded substrate and transparent electrode layers. Example: SLG | FTO.'),
    'ETL_stack_sequence': ('Electron transport layers (ETL)', 'Electron-selective layers, separated by |. Example: TiO2-c | TiO2-mp.'),
    'ETL_deposition_procedure': ('ETL deposition method', 'Recorded deposition procedure for the electron transport layers.'),
    'Perovskite_composition_a_ions': ('A-site ions', 'Recorded A-site labels, separated by ; or |. Example: Cs; FA. No stoichiometry is inferred.'),
    'Perovskite_composition_b_ions': ('B-site ions', 'Recorded B-site labels. Example: Pb; Sn.'),
    'Perovskite_composition_c_ions': ('X-site anions', 'The dataset calls this field c_ions. Example: Br; I.'),
    'Perovskite_composition_short_form': ('Absorber composition', 'Recorded short formula. Keep the ion fields consistent; complex formulas are not automatically validated.'),
    'Perovskite_additives_compounds': ('Additive compounds', 'Recorded additive labels; Undoped means explicitly absent, while missing means not recorded.'),
    'Perovskite_additives_concentrations': ('Additive concentrations', 'Preserve the recorded concentration and units; bare numbers have unspecified units.'),
    'Perovskite_thickness': ('Absorber thickness', 'Example: 500 nm. Bare numbers keep unspecified units; no unit conversion is performed.'),
    'Perovskite_deposition_number_of_deposition_steps': ('Absorber deposition steps', 'Nonnegative integer recording the number of deposition steps.'),
    'Perovskite_deposition_procedure': ('Absorber deposition method', 'Recorded absorber deposition procedure, such as Spin-coating.'),
    'Perovskite_deposition_solvents': ('Deposition solvents', 'Recorded solvent labels. Example: DMF; DMSO.'),
    'Perovskite_deposition_quenching_induced_crystallisation': ('Quenching used', 'Whether quenching-induced crystallisation was reported.'),
    'Perovskite_deposition_thermal_annealing_temperature': ('Annealing temperature', 'Example: 100 C or 373 K. Bare values have unspecified units; negative or ambiguous strings remain raw.'),
    'Perovskite_deposition_thermal_annealing_time': ('Annealing duration', 'Example: 10 min or 600 s. Bare values have unspecified units.'),
    'HTL_stack_sequence': ('Hole transport layers (HTL)', 'Hole-selective layers, separated by |. Example: Spiro-MeOTAD.'),
    'HTL_deposition_procedure': ('HTL deposition method', 'Recorded deposition procedure for the hole transport layers.'),
    'Backcontact_stack_sequence': ('Back contact layers', 'Recorded rear electrode layers. Example: Au or Ag.'),
    'Backcontact_thickness_list': ('Back contact thickness', 'Recorded thicknesses. Example: 90 nm. Bare values have unspecified units.'),
    'Backcontact_deposition_procedure': ('Back contact deposition method', 'Recorded rear electrode deposition procedure, such as Evaporation.'),
    'Encapsulation': ('Device encapsulated', 'Whether encapsulation was reported. Missing differs from No.'),
    'Encapsulation_stack_sequence': ('Encapsulation layers', 'Recorded encapsulation materials. Example: Epoxy | SLG. Unknown does not mean absent.'),
}


def field_label(column):
    return FIELD_GUIDE.get(column, (column.replace('_', ' '), ''))[0]


def _known_parts(value):
    if is_missing(value):
        return []
    return [p.strip().casefold() for p in str(value).split('|')
            if p.strip().casefold() not in ABSENT and not is_missing(p)]


def check_device(record):
    """Return review flags only; do not reject novel devices or infer chemistry."""
    issues = []
    def add(code, fields, message):
        issues.append({'code': code, 'fields': fields, 'message': message})

    stack = _known_parts(record['Cell_stack_sequence'])
    if stack and 'perovskite' in stack:
        absorber = stack.index('perovskite')
        for column in ['Substrate_stack_sequence', 'ETL_stack_sequence', 'HTL_stack_sequence', 'Backcontact_stack_sequence']:
            parts = _known_parts(record[column])
            missing = [part for part in parts if part not in stack]
            if missing:
                add('stack_mismatch', [column, 'Cell_stack_sequence'],
                    f'{field_label(column)} includes layers absent from the full device stack: {", ".join(missing)}. Check the recorded names and layer choices.')
        etl = [stack.index(p) for p in _known_parts(record['ETL_stack_sequence']) if p in stack]
        htl = [stack.index(p) for p in _known_parts(record['HTL_stack_sequence']) if p in stack]
        architecture = str(record['Cell_architecture']).strip().lower()
        wrong = (architecture == 'nip' and (any(i >= absorber for i in etl) or any(i <= absorber for i in htl)))
        wrong |= (architecture == 'pin' and (any(i <= absorber for i in etl) or any(i >= absorber for i in htl)))
        if wrong:
            add('architecture_order', ['Cell_architecture', 'Cell_stack_sequence', 'ETL_stack_sequence', 'HTL_stack_sequence'],
                'The selected nip/pin architecture conflicts with the ETL/absorber/HTL order in the full stack. Check the substrate-to-contact direction.')

    encapsulation = record['Encapsulation']
    material = record['Encapsulation_stack_sequence']
    if encapsulation == 0 and _known_parts(material):
        add('encapsulation_conflict', ['Encapsulation', 'Encapsulation_stack_sequence'],
            'Encapsulation is No, but encapsulation layers are specified. Review these two inputs.')
    if encapsulation == 1 and not is_missing(material) and str(material).strip().lower() in ABSENT:
        add('encapsulation_conflict', ['Encapsulation', 'Encapsulation_stack_sequence'],
            'Encapsulation is Yes, but its layers are explicitly absent.')

    # Only compare simple, unambiguous formulas built from common A/B/X labels.
    formula = record['Perovskite_composition_short_form']
    formula_pattern = r'(?:(?:Cs|FA|MA)(?:\d+(?:\.\d+)?)?)+(?:Pb|Sn)(?:\d+(?:\.\d+)?)?(?:(?:Cl|Br|I)(?:\d+(?:\.\d+)?)?)+'
    if not is_missing(formula) and re.fullmatch(formula_pattern, str(formula)):
        for column, allowed in [('Perovskite_composition_a_ions', {'Cs', 'FA', 'MA'}),
                                ('Perovskite_composition_b_ions', {'Pb', 'Sn'}),
                                ('Perovskite_composition_c_ions', {'Cl', 'Br', 'I'})]:
            selected, incomplete = tokens(record[column])
            present = set(re.findall(r'Cs|FA|MA|Pb|Sn|Cl|Br|I', str(formula))) & allowed
            if selected and not incomplete and set(selected) <= allowed and set(selected) != present:
                add('composition_mismatch', [column, 'Perovskite_composition_short_form'],
                    f'{field_label(column)} does not match the labels in absorber composition {formula}. No concentration or stoichiometry check was performed.')

    for _, (column, units) in MEASURE_FIELDS.items():
        parsed = measurement_summary(record[column], units)
        if parsed['unparsed_count']:
            add('unparsed_measurement', [column], f'{field_label(column)} contains a measurement the numeric parser cannot use; its original text is still passed to the model.')
        if parsed['unitless_count']:
            add('unspecified_units', [column], f'{field_label(column)} contains bare numbers with unspecified units. The model will not assume units.')
    return issues


def explanation_summary(contributions, record, limit=2):
    """Describe signed local contributions without making causal claims."""
    sentences = []
    for positive, verb in [(True, 'raised'), (False, 'lowered')]:
        ranked = sorted(((c, float(v)) for c, v in contributions.items()
                         if (v > 0 if positive else v < 0)), key=lambda pair: abs(pair[1]), reverse=True)[:limit]
        if ranked:
            details = '; '.join(f'{field_label(c)} = {record[c]} ({v:+.2f} PCE points)' for c, v in ranked)
            sentences.append(f'The inputs that most {verb} this model estimate were {details}.')
    if not sentences:
        sentences.append('The supplied inputs did not move this estimate away from the model base value.')
    return ' '.join(sentences) + ' These are model associations, not evidence that changing a material will cause the same efficiency change.'
