"""IESVE -> BPVis, v0.3.1. Run inside IESVE Tools > VEScripts.

SCOPE: monthly CONSUMED energy and hourly Loads_Balance, selectable separately.
Update selected base sheets, or create a new numbered pair in a NEW workbook.
Energy_Balance_Scenarios and all other worksheets remain untouched.
Original worksheet parts remain byte-for-byte when creating a new version;
only the necessary workbook package metadata is extended for the new sheets.
No openpyxl, pandas, Excel installation or pip packages required.

Workflow: choose APS, template, new output file; inspect/review end-use mapping;
then Export or Save diagnostics only. The companion .audit.json records meter
allocation, source identities, unit conversion, errors and reconciliation.
v0.1.1 fixes the hidden mapping dialog with a withdrawn Tk parent, brings it
to the foreground, and records each export stage with visible progress.
v0.1.2 preserves Excel XML namespace declarations, including compatibility
prefixes used only inside attributes. Calendar cells are copied unchanged:
no hour, day, weekday, month or year values are recalculated or shifted.
v0.2.0 reads mapping targets from the selected template. Use Add new column...
in any mapping dropdown to create an additional energy-use column in kWh.
v0.3.0 adds hourly loads in kW, with explicit thermal result selection,
electric end-use/source aggregates, positive PV and hourly net grid surplus.
Calendar cells are preserved byte-for-byte; mismatched years/row counts stop
the export instead of inserting/removing hours or shifting weekdays.
Heating uses the IES hot-water-loop space-conditioning aggregate (review
space-heating scope where HWLs also serve absorption cooling). Cooling uses
chilled-water-loop total load. DHW uses DHW-system demand load. These building
aggregates include the corresponding systems; do not also add their components. Heat rejection must
be mapped to a real thermal result; its auxiliary electricity is not a proxy.
An explicit zero mapping means the user confirms that use is absent.
v0.3.1 adds separate JSON Save mapping / Load mapping buttons in the monthly
and hourly review windows. Saved profiles include custom monthly columns and
hourly source selections, sign conventions and explicit zero choices. Loading
checks the current simulation and flags unresolved selections for review.
The initial screen also offers a new project version: duplicate Energy_Balance
and Loads_Balance as Energy_Balance2 / Loads_Balance2 (then 3, 4, ...), and
write the selected simulation results into those copies. Originals stay intact.

IESVE 2026 reference:
https://help.iesve.com/ve2026/ResultsReader.html
https://help.iesve.com/ve2026/vescripts_editor.html
API integration MUST be validated against VistaPro on the first real export.
This script requires a complete annual APS (365 or 366 days). It does not run
simulations or modify the IES model. At equal 1-hour steps, W / 1000 = kWh per
sample. In general kWh = sum(W) * (24 / results_per_day) / 1000.

Meter policy: direct meter consumption uses add_subs=0. Parent direct use and
child use can then both be included without overlap. For each end use/source,
the difference between the native total and sum of direct named meters is
exported as Unallocated only if nonnegative within numeric tolerance. Any
material excess of meter totals over native totals stops the export.

Header example: Heating__Meter_Name_M12__Electricity_S1_kWh.
All distinct uses mapped to Heating on that meter/source are added together.
Base headers are retained when there is only one contribution for an end use.
Sources/settings outside Energy_Balance remain untouched: check BPVis source
and efficiency assignments when importing. For actual consumption, efficiency
factors normally remain 1 unless intentionally applying a relative change.
"""
from __future__ import annotations

import calendar
import copy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import posixpath
import re
import tempfile
import traceback
from xml.dom import minidom
import xml.etree.ElementTree as ET
from zipfile import ZipFile

VERSION = '0.3.1'
BASE_END_USES = ('Cooling', 'Equipment', 'Heating', 'HotWater', 'Humidification',
                 'Lighting', 'On-site_Generation', 'Other', 'Pumps', 'Ventilation')
# Suggestions only: all USED categories are reviewed before export.
DEFAULT_MAPPING = {
    'prm_space_cooling': 'Cooling', 'prm_space_heating': 'Heating',
    'prm_services_water_heating': 'HotWater', 'prm_humidification': 'Humidification',
    'prm_pumps': 'Pumps', 'prm_elec_gen_pv': 'On-site_Generation',
    'prm_interior_lighting': 'Lighting', 'prm_exterior_lighting': 'Lighting',
    'prm_interior_lighting_process': 'Lighting', 'prm_interior_lighting_unregulated': 'Lighting',
    'prm_receptacle_equipment': 'Equipment', 'prm_data_center_equipment': 'Equipment',
    'prm_cooking': 'Equipment', 'prm_refrigeration': 'Equipment',
    'prm_elevators_escalators': 'Equipment',
    'prm_fans_interior_central': 'Ventilation', 'prm_fans_interior_local': 'Ventilation',
    'prm_fans_exhaust': 'Ventilation', 'prm_fans_garage': 'Ventilation',
    'prm_fans_process': 'Ventilation', 'prm_heat_rejection': 'Other',
    'prm_other_process': 'Other', 'prm_motor': 'Other', 'prm_transformer': 'Other',
    # Keep non-PV generation separate, not silently folded into PV.
    'prm_elec_gen_chp': 'CHP_Generation', 'prm_elec_gen_wind': 'Wind_Generation',
}
GENERATION_ENUMS = {'prm_elec_gen_pv', 'prm_elec_gen_chp', 'prm_elec_gen_wind'}
GENERATION_TARGETS = {'On-site_Generation', 'CHP_Generation', 'Wind_Generation'}
ADD_NEW_COLUMN = 'Add new column...'
NS = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
ET.register_namespace('', NS)


class ExportError(RuntimeError):
    pass


def finite_number(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def id_key(value):
    try:
        return str(int(value))
    except (ValueError, TypeError):
        return str(value)


def metadata_entries(raw):
    if not isinstance(raw, dict):
        raise ExportError('Expected a dictionary from IES metadata API.')
    result = []
    for key, value in raw.items():
        if not isinstance(value, dict):
            raise ExportError('Unexpected metadata item: ' + repr(value))
        item = dict(value)
        item.setdefault('id', key)
        # Use original IES enum/id objects in native calls, not constructed enums.
        if id_key(item['id']) != '0' and item.get('used', True):
            result.append(item)
    return sorted(result, key=lambda x: id_key(x['id']))


def enum_name(api, use_id):
    for name in dir(api.EnergyUse):
        if name.startswith('prm_') and id_key(getattr(api.EnergyUse, name)) == id_key(use_id):
            return name
    return str(getattr(use_id, 'name', ''))


def inspect_results(reader, api):
    uses = metadata_entries(reader.get_energy_uses(used_only=True))
    for use in uses:
        use['enum_name'] = enum_name(api, use['id'])
    sources = metadata_entries(reader.get_energy_sources(used_only=True))
    meters = metadata_entries(reader.get_energy_meters(used_only=True))
    info = {}
    for field in ('first_day', 'last_day', 'results_per_day', 'simulation_year',
                  'plot_year', 'first_weekday_plot_year', 'plot_data_offset_secs',
                  'weather_file', 'hvac_file'):
        info[field] = getattr(reader, field, None)
    return uses, sources, meters, info


def check_calendar(info, year):
    first, last, per_day = (int(info[k]) for k in ('first_day', 'last_day', 'results_per_day'))
    # IES has a separate "Treat simulation as leap year" option. Full-year
    # APS day count, rather than the nominal simulation year, is authoritative.
    days = last
    if first != 1 or days not in (365, 366):
        raise ExportError('This first version requires a full-year APS, day 1 through 365 or 366; '
                          'this file covers day %s to %s.' % (first, last))
    if per_day <= 0:
        raise ExportError('Invalid APS results_per_day.')
    return days * per_day, per_day


def vector(raw, count, context):
    if raw is None:
        return None
    try:
        values = [float(x) for x in raw]
    except (TypeError, ValueError) as exc:
        raise ExportError('Invalid array for ' + context) from exc
    if len(values) != count or not all(math.isfinite(x) for x in values):
        raise ExportError('%s: expected %s finite samples, received %s.' % (context, count, len(values)))
    return values


def read_power(reader, count, use_id, source_id=None, meter_id=None, direct=False):
    args = dict(use_id=use_id, type='e', start_day=int(reader.first_day), end_day=int(reader.last_day))
    if source_id is not None:
        args['source_id'] = source_id
    if meter_id is not None:
        args['meter_id'] = meter_id
    args['add_subs'] = 0 if direct else -1
    context = 'use=%s source=%s meter=%s' % (use_id, source_id, meter_id)
    try:
        return vector(reader.get_energy_results(**args), count, context)
    except Exception as exc:
        raise ExportError('IES read failed (%s). %s' % (context, exc)) from exc


def sign_normalized(values, generation, context):
    if values is None:
        return None
    # APS generation can be represented with either sign. Keep one convention
    # for reconciliation, then write generation as a negative offset in Excel.
    sign = -1 if generation and math.fsum(values) < 0 else 1
    values = [v * sign for v in values]
    tolerance = max(0.1, max((abs(v) for v in values), default=0) * 1e-5)
    if min(values, default=0) < -tolerance:
        raise ExportError('Unexpected negative/mixed-sign consumption in ' + context + '. Review the result definition.')
    return [max(0.0, v) for v in values]


def monthly_kwh(values, year, per_day):
    days, remainder = divmod(len(values), per_day)
    if remainder or days not in (365, 366):
        raise ExportError('Monthly integration requires a full 365/366-day array.')
    calendar_year = 2000 if days == 366 else 2001  # month lengths only
    result, cursor = [], 0
    multiplier = 24.0 / per_day / 1000.0
    for month in range(1, 13):
        stop = cursor + calendar.monthrange(calendar_year, month)[1] * per_day
        result.append(math.fsum(values[cursor:stop]) * multiplier)
        cursor = stop
    if cursor != len(values):
        raise ExportError('Calendar/sample mismatch during monthly integration.')
    return result


def tolerance(a, b):
    # Raw arrays are float32 in the documented API: 0.1 W or 0.01%.
    return max(0.1, abs(a) * 1e-4, abs(b) * 1e-4)


def reconcile_equal(expected, actual, context):
    failures = [i for i, (a, b) in enumerate(zip(expected, actual)) if abs(a-b) > tolerance(a, b)]
    if failures:
        i = failures[0]
        raise ExportError('%s does not reconcile at sample %s: %.6g vs %.6g W. Export stopped.'
                          % (context, i + 1, expected[i], actual[i]))


def slug(text):
    out = re.sub(r'[^\w]+', '_', str(text), flags=re.UNICODE).strip('_')
    return out or 'Unnamed'


def extract_energy(reader, api, uses, sources, meters, info, mapping, year, audit, progress=print,
                   output_targets=None):
    count, per_day = check_calendar(info, year)
    if not uses or not sources:
        raise ExportError('No energy end uses or sources exposed by this APS file.')
    groups = {}
    target_types = {}
    source_lookup = {id_key(x['id']): x for x in sources}
    audit['unit_conversion'] = {'input': 'W (get_energy_results type=e)', 'output': 'kWh',
                                'hours_per_sample': 24.0 / per_day, 'simulation_year': year,
                                'days_in_calendar': count // per_day, 'leap_calendar': count // per_day == 366,
                                'grouping': 'APS day order; results_per_day samples per day',
                                'plot_offset': 'recorded only; no artificial shift of APS day bins'}
    audit['reconciliation'] = []
    audit['allocations'] = []
    for use in uses:
        uid = use['id']; label = str(use.get('name', uid)); target = mapping[id_key(uid)]
        if not target or target in ('UNASSIGNED', ADD_NEW_COLUMN):
            raise ExportError('Please assign the used category: ' + label)
        generation = use['enum_name'] in GENERATION_ENUMS
        if target.casefold() in {v.casefold() for v in GENERATION_TARGETS} and not generation:
            raise ExportError('Only generation end uses may map to a generation category: ' + label)
        if generation and target.casefold() in {v.casefold() for v in BASE_END_USES if v not in GENERATION_TARGETS}:
            raise ExportError('Map generation to a generation category or add a separate new column: ' + label)
        if target.casefold() in target_types and target_types[target.casefold()] != generation:
            raise ExportError('Consumption and generation cannot share the target column ' + target)
        target_types[target.casefold()] = generation
        progress('Reading ' + label + ' ...')
        all_total = sign_normalized(read_power(reader, count, uid), generation, label)
        if all_total is None:
            raise ExportError('IES marks this end use as used but returns no aggregate: ' + label)
        direct_by_source = {}
        # Query each named meter's DIRECT contribution. Never add inclusive
        # parent totals to their children.
        for meter in meters:
            mid = meter['id']
            sid = id_key(meter.get('source_id'))
            if sid not in source_lookup:
                raise ExportError('Used meter %s has an unknown/missing source (%s).' % (mid, sid))
            raw = read_power(reader, count, uid, source_id=source_lookup[sid]['id'], meter_id=mid, direct=True)
            direct = sign_normalized(raw, generation, label + ' / ' + str(meter.get('name', mid)))
            if direct is None or not any(direct):
                continue
            sid = id_key(meter.get('source_id'))
            if sid not in source_lookup:
                raise ExportError('Nonzero meter %s has an unknown/missing energy source (%s).'
                                  % (meter.get('name', mid), sid))
            direct_by_source.setdefault(sid, []).append((meter, direct))
        source_sum = [0.0] * count
        for source in sources:
            sid = id_key(source['id'])
            total = sign_normalized(read_power(reader, count, uid, source_id=source['id']), generation, label)
            entries = direct_by_source.get(sid, [])
            if total is None:
                if entries:
                    raise ExportError('Missing source aggregate for a nonzero meter: ' + label)
                continue
            source_sum = [a+b for a,b in zip(source_sum, total)]
            allocated = [0.0] * count
            for meter, values in entries:
                allocated = [a+b for a,b in zip(allocated, values)]
            residual = []
            for i, (t, a) in enumerate(zip(total, allocated)):
                if a - t > tolerance(t, a):
                    raise ExportError('Meter double-counting/inconsistent add_subs=0 for %s / %s at sample %s: '
                                      'meters %.6g W > total %.6g W.' % (label, source.get('name', sid), i+1, a, t))
                # Keep even small positive amounts: they can add up materially
                # over a year. Only clamp tolerated negative round-off.
                residual.append(max(0.0, t-a))
            if any(residual):
                entries = entries + [({'id': 'unallocated', 'name': 'Unallocated'}, residual)]
            resolved = [a+b for a,b in zip(allocated, residual)]
            reconcile_equal(total, resolved, label + ' meter partition')
            expected_kwh = math.fsum(total) * 24 / per_day / 1000
            actual_kwh = math.fsum(resolved) * 24 / per_day / 1000
            if abs(actual_kwh - expected_kwh) > max(0.01, abs(expected_kwh) * 0.0001):
                raise ExportError('Annual reconciliation failed for ' + label)
            audit['reconciliation'].append({'use': label, 'source': str(source.get('name', sid)),
                'native_kwh': expected_kwh, 'allocated_kwh': actual_kwh, 'status': 'PASS'})
            for meter, values in entries:
                mid = id_key(meter['id']); sign = -1 if generation else 1
                group_key = (target, mid, sid)
                months = [sign*v for v in monthly_kwh(values, year, per_day)]
                if group_key not in groups:
                    groups[group_key] = {'months': [0.0]*12, 'meter': str(meter.get('name', mid)),
                                         'source': str(source.get('name', sid)), 'uses': [], 'generation': generation}
                group = groups[group_key]
                if group['generation'] != generation:
                    raise ExportError('Consumption and generation cannot share one output column.')
                group['months'] = [a+b for a,b in zip(group['months'], months)]
                group['uses'].append(label)
                audit['allocations'].append({'use': label, 'target': target, 'meter_id': mid,
                    'meter': group['meter'], 'source_id': sid, 'source': group['source'],
                    'annual_kwh': math.fsum(months)})
        reconcile_equal(all_total, source_sum, label + ' source partition')
        native_kwh = math.fsum(all_total) * 24 / per_day / 1000
        source_kwh = math.fsum(source_sum) * 24 / per_day / 1000
        if abs(source_kwh - native_kwh) > max(0.01, abs(native_kwh) * 0.0001):
            raise ExportError('Annual source reconciliation failed for ' + label)
        audit['reconciliation'].append({'use': label, 'source': 'ALL_SOURCES',
            'native_kwh': native_kwh, 'allocated_kwh': source_kwh, 'status': 'PASS'})
    columns = []
    categories = list(BASE_END_USES if output_targets is None else output_targets)
    for use in uses:
        target = mapping[id_key(use['id'])]
        if target not in categories:
            categories.append(target)
    if len({v.casefold() for v in categories}) != len(categories):
        raise ExportError('Duplicate target column names (case-insensitive).')
    for target in categories:
        matching = [(key, val) for key, val in groups.items() if key[0] == target]
        if not matching:
            columns.append((target+'_kWh', [0.0]*12))
            continue
        for (_, mid, sid), val in sorted(matching, key=lambda pair: pair[0]):
            if len(matching) == 1:
                header = target + '_kWh'
            else:
                header = '%s__%s_M%s__%s_S%s_kWh' % (target, slug(val['meter']), slug(mid), slug(val['source']), slug(sid))
            columns.append((header, val['months']))
            val['header'] = header
    if len({h.casefold() for h,_ in columns}) != len(columns):
        raise ExportError('Output header collision.')
    audit['output_target_order'] = categories
    audit['columns'] = [{'header': v['header'], 'source': v['source'], 'meter': v['meter'],
                         'ies_uses': v['uses'], 'monthly_kwh': v['months']} for v in groups.values()]
    audit['notes'] = [
        'Energy_Balance contains consumed energy, not thermal loads.',
        'Monthly extraction writes Energy_Balance; hourly export is controlled separately. Energy_Balance_Scenarios is untouched.',
        'Existing calendar cells are preserved. No hour, day, weekday, month or year fields are shifted or recalculated.',
        'Check BPVis source/efficiency assignments for split columns. Mapping is recorded here, not written to settings sheets.',
        'Unallocated represents the verified source/use total minus named DIRECT meter results.',
        'On-site_Generation is negative generation; all meter/source partitions reconciled before aggregation.',
        'Compare this first export with VistaPro monthly totals before project use.'
    ]
    return columns


def excel_column(number):
    text = ''
    while number:
        number, digit = divmod(number-1, 26)
        text = chr(65+digit)+text
    return text


def target_sheet_path(z):
    workbook = ET.fromstring(z.read('xl/workbook.xml'))
    relations = {r.attrib['Id']: r.attrib['Target'] for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
    for sheet in workbook.findall('{%s}sheets/{%s}sheet' % (NS, NS)):
        if sheet.attrib['name'] == 'Energy_Balance':
            target = relations[sheet.attrib['{%s}id' % REL]]
            return target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/'+target)
    raise ExportError('Template has no exact Energy_Balance worksheet.')


def validate_markup_namespaces(document):
    """Check namespace references inside Office compatibility attribute values."""
    mc = 'http://schemas.openxmlformats.org/markup-compatibility/2006'
    for element in document.getElementsByTagName('*'):
        references = []
        for name in ('Ignorable', 'MustUnderstand'):
            if element.hasAttributeNS(mc, name):
                references.extend(element.getAttributeNS(mc, name).split())
        for name in ('ProcessContent', 'PreserveElements', 'PreserveAttributes'):
            if element.hasAttributeNS(mc, name):
                references.extend(v.split(':', 1)[0] for v in element.getAttributeNS(mc, name).split() if ':' in v)
        if element.namespaceURI == mc and element.localName == 'Choice':
            references.extend(element.getAttribute('Requires').split())
        for prefix in references:
            if prefix == 'xml':
                continue
            current = element
            while current.nodeType == current.ELEMENT_NODE:
                if current.hasAttribute('xmlns:' + prefix) and current.getAttribute('xmlns:' + prefix):
                    break
                current = current.parentNode
            else:
                raise ExportError('Unbound Excel compatibility namespace: %s. '
                                  'Use an undamaged template or the repaired workbook.' % prefix)


def build_sheet_xml(original, columns):
    # ElementTree rewrites prefixes and drops unused namespace declarations,
    # but does not rewrite QName-valued attributes such as mc:Ignorable. Keep
    # Excel's original declarations and prefixes intact with the DOM instead.
    with minidom.parseString(original) as doc:
        validate_markup_namespaces(doc)
        root = doc.documentElement
        if root.namespaceURI != NS or root.localName != 'worksheet':
            raise ExportError('Unexpected worksheet XML format.')
        def children(node, name):
            return [n for n in node.childNodes if n.nodeType == n.ELEMENT_NODE
                    and n.namespaceURI == NS and n.localName == name]
        for tag in ('tableParts', 'mergeCells'):
            if children(root, tag):
                raise ExportError('Energy_Balance contains %s; use the supplied plain-grid template.' % tag)
        sheets = children(root, 'sheetData')
        if len(sheets) != 1:
            raise ExportError('Energy_Balance sheetData missing or duplicated.')
        old = sheets[0]
        rows = {row.getAttribute('r'): row for row in children(old, 'row')}
        cells = {c.getAttribute('r'): c for row in rows.values() for c in children(row, 'c')}
        new = old.cloneNode(False)  # Retain any locally declared namespaces.
        def element(name):
            return doc.createElementNS(NS, (old.prefix + ':' if old.prefix else '') + name)
        for r in range(1, 14):
            calendar_cell = cells.get('A' + str(r))
            if calendar_cell is None:
                raise ExportError('Template calendar cell A%s is missing. Calendar fields are never generated.' % r)
            row = rows[str(r)].cloneNode(False)
            # Calendar column is kept verbatim, including shared-string IDs,
            # numeric month labels, formulas, styles and existing text.
            row.appendChild(calendar_cell.cloneNode(True))
            values = [h for h, _ in columns] if r == 1 else [v[r-2] for _, v in columns]
            for col, value in enumerate(values, 2):
                ref = excel_column(col) + str(r)
                old_cell = cells.get(ref)
                if old_cell is None:
                    old_cell = cells.get('B' + str(r))
                cell = element('c'); cell.setAttribute('r', ref)
                if old_cell is not None and old_cell.hasAttribute('s'):
                    cell.setAttribute('s', old_cell.getAttribute('s'))
                if isinstance(value, str):
                    cell.setAttribute('t', 'inlineStr')
                    inline = element('is'); text = element('t')
                    text.appendChild(doc.createTextNode(value))
                    inline.appendChild(text); cell.appendChild(inline)
                else:
                    if not finite_number(value):
                        raise ExportError('Non-finite output at ' + ref)
                    numeric = element('v')
                    numeric.appendChild(doc.createTextNode(format(float(value), '.15g')))
                    cell.appendChild(numeric)
                row.appendChild(cell)
            new.appendChild(row)
        root.replaceChild(new, old)
        for tag in ('dimension', 'autoFilter'):
            for item in children(root, tag):
                item.setAttribute('ref', 'A1:%s13' % excel_column(len(columns)+1))
        updated = doc.toxml(encoding='UTF-8')
    with minidom.parseString(updated) as check:
        validate_markup_namespaces(check)
    return updated




def json_safe(value):
    if isinstance(value,dict):
        return {str(k): json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [json_safe(v) for v in value]
    if value is None or isinstance(value,(str,int,float,bool)):
        return value
    return str(value)


def write_audit(path, audit):
    with open(path,'w',encoding='utf-8') as stream:
        json.dump(json_safe(audit),stream,ensure_ascii=False,indent=2,allow_nan=False)


def new_column_target(name, existing_targets):
    """Validate a user-entered name without silently changing its spelling."""
    if not isinstance(name, str):
        raise ExportError('Enter a name for the new energy-use column.')
    if any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF or ord(c) in (0xFFFE, 0xFFFF) for c in name):
        raise ExportError('Column names must be one line and cannot contain control characters.')
    target = name.strip()
    if target.casefold().endswith('_kwh'):
        target = target[:-4].rstrip()
    if not target or target.casefold().endswith('_kwh'):
        raise ExportError('Enter an energy-use name, with at most one optional _kWh suffix.')
    if len(target) > 32763:
        raise ExportError('The column name is too long for Excel.')
    if target.casefold() in {'unassigned', ADD_NEW_COLUMN.casefold()}:
        raise ExportError('That name is reserved for the mapping controls.')
    if target.casefold() in {v.casefold() for v in existing_targets}:
        raise ExportError('That column already exists. Select it from the dropdown.')
    return target


def read_energy_template_columns(template):
    """Read actual Energy_Balance header names, including shared-string cells."""
    with ZipFile(template) as workbook:
        raw = workbook.read(target_sheet_path(workbook))
        with minidom.parseString(raw) as document:
            validate_markup_namespaces(document)
        root = ET.fromstring(raw)
        shared = []
        if 'xl/sharedStrings.xml' in workbook.namelist():
            shared_root = ET.fromstring(workbook.read('xl/sharedStrings.xml'))
            shared = [''.join(node.text or '' for node in item.iter('{%s}t' % NS))
                      for item in shared_root.findall('{%s}si' % NS)]
        header = root.find('{%s}sheetData/{%s}row[@r="1"]' % (NS, NS))
        if header is None:
            raise ExportError('Energy_Balance must have its column headers in row 1.')
        headers = {}
        for cell in header.findall('{%s}c' % NS):
            ref = cell.get('r', '')
            match = re.fullmatch(r'([A-Z]+)1', ref)
            if not match:
                raise ExportError('Invalid Energy_Balance header cell reference: ' + ref)
            kind = cell.get('t')
            if kind == 'inlineStr':
                value = ''.join(node.text or '' for node in cell.iter('{%s}t' % NS))
            else:
                value = cell.findtext('{%s}v' % NS, default='')
                if kind == 's':
                    try:
                        index = int(value)
                        if index < 0:
                            raise ValueError('negative index')
                        value = shared[index]
                    except (ValueError, IndexError) as exc:
                        raise ExportError('Invalid shared-string header in ' + ref) from exc
            if not value:
                continue  # Ignore purely styled, unused header cells.
            col = 0
            for letter in match.group(1):
                col = 26 * col + ord(letter) - ord('A') + 1
            headers[col] = value
        if headers.get(1, '').strip().casefold() != 'month':
            raise ExportError('Expected the existing Month calendar column in Energy_Balance!A1.')
        if max(headers, default=0) < 2 or set(headers) != set(range(1, max(headers) + 1)):
            raise ExportError('Energy_Balance needs contiguous energy-use headers after Month.')
        targets = []
        for col in range(2, max(headers) + 1):
            title = headers[col]
            if not title.endswith('_kWh'):
                raise ExportError('Energy_Balance column %s must end with _kWh: %s' % (excel_column(col), title))
            target = new_column_target(title, targets)
            if target + '_kWh' != title:
                raise ExportError('Energy_Balance header has extra whitespace or suffixes: ' + title)
            targets.append(target)
        return targets


def show_window(win):
    """Map an independent window before grabbing input (also in embedded IES)."""
    win.update_idletasks()
    win.deiconify()
    win.lift()
    # Temporarily raise above the IES host, without leaving it always on top.
    try:
        win.attributes('-topmost', True)
        # Own the timer on the persistent root. The review window can close
        # before one second; destroying it must not delete a pending Tcl command.
        timer_owner = win
        while getattr(timer_owner, 'master', None) is not None:
            timer_owner = timer_owner.master
        timer_owner.after(1000, lambda: win.attributes('-topmost', False) if win.winfo_exists() else None)
    except Exception:
        pass  # Some window managers do not support topmost.
    if not win.winfo_viewable():
        win.wait_visibility()
    win.focus_force()


"""Versioned, portable monthly/hourly JSON mapping profiles (stdlib only).

Host supplies ExportError, VERSION, id_key, json_safe, new_column_target,
_loads_validate_mapping and _loads_new_hvac_term. No results or workbook data
are stored in profiles, and no file-supplied Python is evaluated.
"""

MAPPING_PROFILE_FORMAT = 'bpvis_iesve_mapping'
MAPPING_PROFILE_SCHEMA_VERSION = 1
MAPPING_PROFILE_MAX_BYTES = 5 * 1024 * 1024
MAPPING_PROFILE_TYPES = ('monthly_energy', 'hourly_loads')


def _mapping_profile_envelope(profile, expected_type=None):
    if not isinstance(profile, dict) or profile.get('format') != MAPPING_PROFILE_FORMAT:
        raise ExportError('This is not a BPVis IESVE mapping file.')
    version = profile.get('schema_version')
    if type(version) is not int or version != MAPPING_PROFILE_SCHEMA_VERSION:
        raise ExportError('Unsupported mapping-file schema version: %r.' % version)
    kind = profile.get('mapping_type')
    if kind not in MAPPING_PROFILE_TYPES:
        raise ExportError('Unknown mapping-file type: %r.' % kind)
    if expected_type is not None and kind != expected_type:
        readable = {'monthly_energy': 'monthly Energy Balance', 'hourly_loads': 'hourly Loads Balance'}
        raise ExportError('This file contains a %s mapping. Load a %s mapping in this window.' %
                          (readable[kind], readable.get(expected_type, expected_type)))
    return profile


def _new_mapping_profile(kind):
    import datetime
    return {'format': MAPPING_PROFILE_FORMAT, 'schema_version': MAPPING_PROFILE_SCHEMA_VERSION,
            'mapping_type': kind, 'exporter_version': VERSION,
            'created_at': datetime.datetime.now().isoformat(timespec='seconds')}


def read_mapping_profile(path, expected_type):
    import json
    from pathlib import Path
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def constant(value):
        raise ValueError('Non-finite JSON number: ' + value)
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError('Non-finite JSON number: ' + value)
        return number
    try:
        with Path(path).open('rb') as stream:
            raw = stream.read(MAPPING_PROFILE_MAX_BYTES + 1)
        if len(raw) > MAPPING_PROFILE_MAX_BYTES:
            raise ExportError('Mapping file is larger than 5 MiB.')
        profile = json.loads(raw.decode('utf-8-sig'), object_pairs_hook=pairs,
                             parse_constant=constant, parse_float=finite_float)
        return _mapping_profile_envelope(profile, expected_type)
    except ExportError:
        raise
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError) as exc:
        raise ExportError('Cannot read mapping JSON: ' + str(exc)) from exc


def write_mapping_profile(path, profile):
    import json
    import os
    import tempfile
    from pathlib import Path
    _mapping_profile_envelope(profile)
    if Path(path).suffix.casefold() != '.json':
        raise ExportError('Save mapping files with a .json extension.')
    temporary = None
    try:
        encoded = (json.dumps(json_safe(profile), ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
        if len(encoded) > MAPPING_PROFILE_MAX_BYTES:
            raise ExportError('Mapping file is larger than 5 MiB.')
        destination = Path(path)
        fd, temporary = tempfile.mkstemp(prefix='.bpvis_mapping_', suffix='.json', dir=str(destination.parent))
        with os.fdopen(fd, 'wb') as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
    except ExportError:
        raise
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        raise ExportError('Cannot save mapping JSON: ' + str(exc)) from exc
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def _profile_string(value, label, allow_empty=False):
    if not isinstance(value, str) or (not value and not allow_empty):
        raise ExportError(label + ' must be text' + ('.' if allow_empty else ' and cannot be empty.'))
    if any(ord(c) < 32 for c in value):
        raise ExportError(label + ' cannot contain control characters.')
    return value


def _profile_targets(targets):
    if not isinstance(targets, list) or not targets:
        raise ExportError('The monthly mapping must contain its energy-use target columns.')
    result = []
    for target in targets:
        normalized = new_column_target(target, result)
        if normalized != target:
            raise ExportError('Mapping target names must omit the _kWh suffix and surrounding whitespace.')
        result.append(target)
    return result


def _profile_use_identity(use):
    if not isinstance(use, dict) or 'id' not in use:
        raise ExportError('Each monthly assignment needs an IES use identity.')
    uid = _profile_string(str(use['id']) if isinstance(use['id'], int) and not isinstance(use['id'], bool) else use['id'], 'IES use ID')
    enum = _profile_string(use.get('enum_name', ''), 'IES use enum', allow_empty=True)
    name = _profile_string(use.get('name', ''), 'IES use name', allow_empty=True)
    if not enum and not name:
        raise ExportError('An IES use without an enum needs its name as well as its ID.')
    return {'id': uid, 'enum_name': enum, 'name': name}


def make_energy_mapping_profile(uses, targets, mapping):
    targets = _profile_targets(list(targets))
    assignments = []
    for use in uses:
        identity = {'id': id_key(use['id']), 'enum_name': str(use.get('enum_name') or ''),
                    'name': str(use.get('name') or '')}
        target = mapping.get(identity['id'], 'UNASSIGNED')
        if target != 'UNASSIGNED' and target not in targets:
            raise ExportError('Unknown monthly target for ' + identity['name'])
        assignments.append({'ies_use': identity, 'target': target})
    profile = _new_mapping_profile('monthly_energy')
    profile.update(targets=targets, assignments=assignments)
    # Reuse full import validation to reject duplicate identities/targets.
    apply_energy_mapping_profile(profile, uses, targets)
    return profile


def apply_energy_mapping_profile(profile, uses, template_targets):
    _mapping_profile_envelope(profile, 'monthly_energy')
    saved_targets = _profile_targets(profile.get('targets'))
    targets = _profile_targets(list(template_targets))
    by_case = {target.casefold(): target for target in targets}
    added = []
    for target in saved_targets:
        if target.casefold() not in by_case:
            targets.append(target); added.append(target); by_case[target.casefold()] = target
    saved_target_set = set(saved_targets)
    assignments = profile.get('assignments')
    if not isinstance(assignments, list):
        raise ExportError('Monthly assignments must be a list.')
    current = [{**_profile_use_identity({'id': id_key(use['id']), 'enum_name': str(use.get('enum_name') or ''),
                    'name': str(use.get('name') or '')})} for use in uses]
    if len({use['id'] for use in current}) != len(current):
        raise ExportError('Duplicate current IES use IDs.')
    mapping = {use['id']: 'UNASSIGNED' for use in current}
    warnings, identities, resolved_ids = [], set(), set()
    for assignment in assignments:
        if not isinstance(assignment, dict):
            raise ExportError('Invalid monthly assignment.')
        identity = _profile_use_identity(assignment.get('ies_use'))
        signature = ('enum', identity['enum_name']) if identity['enum_name'] else ('id_name', identity['id'], identity['name'])
        if signature in identities:
            raise ExportError('The mapping contains a duplicate IES use: ' + (identity['name'] or identity['enum_name']))
        identities.add(signature)
        target = assignment.get('target')
        if not isinstance(target, str) or (target != 'UNASSIGNED' and target not in saved_target_set):
            raise ExportError('A monthly assignment references an undefined target column.')
        matches = [use for use in current if (use['enum_name'] == identity['enum_name'] if identity['enum_name']
                   else use['id'] == identity['id'] and use['name'] == identity['name'])]
        if len(matches) != 1:
            warnings.append('Saved IES use unavailable or ambiguous: %s. Its assignment was not applied.' %
                            (identity['name'] or identity['enum_name']))
            continue
        uid = matches[0]['id']
        if uid in resolved_ids:
            raise ExportError('More than one saved assignment resolves to the same current IES use.')
        resolved_ids.add(uid)
        mapping[uid] = target if target == 'UNASSIGNED' else by_case[target.casefold()]
    for use in current:
        if use['id'] not in resolved_ids:
            warnings.append('Current IES use has no saved assignment: %s. Select a target before export.' % (use['name'] or use['id']))
    return {'targets': targets, 'added_targets': added, 'mapping': mapping, 'warnings': warnings}


def _profile_load_choice(item, target):
    if not isinstance(item, dict):
        raise ExportError('Invalid hourly mapping for ' + target)
    mode = item.get('mode')
    if mode not in ('sum', 'zero', 'unassigned', 'grid'):
        raise ExportError('Invalid hourly mapping mode for ' + target)
    sign = item.get('sign', 'positive')
    if sign not in ('positive', 'negative', 'magnitude'):
        raise ExportError('Invalid sign convention for ' + target)
    terms = item.get('terms', [])
    if not isinstance(terms, list) or any(not isinstance(key, str) or not key for key in terms):
        raise ExportError('Hourly result selections must be a list of term IDs.')
    if len(terms) != len(set(terms)):
        raise ExportError('Duplicate hourly result in ' + target)
    if mode == 'sum' and not terms:
        raise ExportError('Empty sum mapping for ' + target)
    if mode != 'sum' and terms:
        raise ExportError('Only a sum mapping can contain selected results: ' + target)
    if (target == 'Grid_Injection') != (mode == 'grid'):
        raise ExportError('Grid_Injection must use the calculated grid mode, which cannot be assigned to other columns.')
    result = {'mode': mode, 'terms': list(terms), 'sign': sign}
    if 'note' in item:
        note = item['note']
        if not isinstance(note, str):
            raise ExportError('Mapping notes must be text.')
        result['note'] = note
    if 'explicit_zero' in item:
        if not isinstance(item['explicit_zero'], bool):
            raise ExportError('explicit_zero must be true or false.')
        result['explicit_zero'] = item['explicit_zero']
    return result


def _profile_load_term(term):
    import math
    if not isinstance(term, dict):
        raise ExportError('Invalid hourly result descriptor.')
    kind = term.get('kind')
    if kind not in ('model', 'source', 'energy_use', 'apache_system', 'hvac_component'):
        raise ExportError('Unsupported hourly result descriptor type: %r.' % kind)
    _profile_string(term.get('label'), 'Result label')
    try:
        factor = float(term['unit_factor_to_kw'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ExportError('A saved result has an invalid power-unit conversion.') from exc
    if not math.isfinite(factor) or factor <= 0:
        raise ExportError('A saved result has an invalid power-unit conversion.')
    if term.get('quantity') not in ('thermal', 'electricity', 'energy'):
        raise ExportError('A saved result has an invalid quantity.')
    if kind in ('model', 'apache_system'):
        for field in ('aps_var', 'display_name', 'model_level'):
            _profile_string(term.get(field), 'Result ' + field)
        if term['model_level'] not in (('l', 'e') if kind == 'model' else ('v',)):
            raise ExportError('Unsupported model level in saved result.')
        if kind == 'apache_system':
            _profile_string(term.get('system_id'), 'Apache System ID')
    if kind in ('source', 'energy_use'):
        if kind == 'energy_use':
            _profile_string(id_key(term.get('use_id')) if term.get('use_id') is not None else None, 'Energy use ID')
            _profile_string(term.get('use_enum', ''), 'Energy use enum', allow_empty=True)
        if kind == 'source' or term.get('source_id') is not None:
            _profile_string(id_key(term.get('source_id')) if term.get('source_id') is not None else None, 'Source ID')
            _profile_string(term.get('source_enum', ''), 'Source enum', allow_empty=True)
    if kind == 'hvac_component':
        _profile_string(term.get('component_id'), 'HVAC component ID')
        _profile_string(term.get('var_name'), 'HVAC result name')
        if type(term.get('component_type')) is not int or term['component_type'] < 0:
            raise ExportError('Saved HVAC component type must be a nonnegative integer.')
        if factor not in (0.001, 1.0, 1000.0) or term.get('quantity') != 'thermal':
            raise ExportError('Saved HVAC component result needs a supported thermal power unit.')
    return term


def _profile_headers(headers):
    if not isinstance(headers, list) or not headers:
        raise ExportError('Hourly mapping columns must be a nonempty list.')
    if any(not isinstance(header, str) or not header for header in headers) or len(headers) != len(set(headers)):
        raise ExportError('Hourly mapping columns must be unique nonempty names.')
    calendar = {'hoy', 'doy', 'day', 'month', 'weekday', 'hour', 'year'}
    if any(header.strip().casefold() in calendar for header in headers):
        raise ExportError('Calendar fields cannot be included in load mappings.')
    return list(headers)


def _validate_partial_load_mapping(mapping, catalog):
    # Drafts may contain unresolved columns; assigned columns must still obey
    # all quantity, overlap and Equipment/Other rules before a profile applies.
    assigned = {key: value for key, value in mapping.items() if value['mode'] != 'unassigned'}
    if assigned:
        _loads_validate_mapping(assigned, list(assigned), catalog)


def make_loads_mapping_profile(mapping, catalog, headers):
    headers = _profile_headers(list(headers))
    choices = {target: _profile_load_choice(mapping.get(target, {'mode': 'unassigned', 'terms': [], 'sign': 'positive'}), target)
               for target in headers}
    selected = {key for item in choices.values() for key in item['terms']}
    missing = selected - set(catalog)
    if missing:
        raise ExportError('A selected result is absent from the current catalog.')
    terms = {}
    for key in sorted(selected):
        descriptor = dict(_profile_load_term(catalog[key]))
        # IES enum objects are retained in the runtime catalog. Persist their
        # stable textual IDs rather than relying on an object's repr().
        for field in ('use_id', 'source_id'):
            if descriptor.get(field) is not None:
                descriptor[field] = id_key(descriptor[field])
        terms[key] = json_safe(descriptor)
    _validate_partial_load_mapping(choices, catalog)
    profile = _new_mapping_profile('hourly_loads')
    profile.update(columns=headers, mapping=choices, results=terms)
    return profile


def _profile_enum_or_id_match(saved, live, enum_field, id_field, label_fallback=True):
    if saved.get(enum_field):
        return live.get(enum_field) == saved[enum_field]
    if saved.get(id_field) is None or live.get(id_field) is None:
        return saved.get(id_field) is None and live.get(id_field) is None
    return id_key(saved[id_field]) == id_key(live[id_field]) and (not label_fallback or saved['label'] == live.get('label'))


def _profile_match_term(saved, catalog):
    kind = saved['kind']
    matches = []
    for key, live in catalog.items():
        if live.get('kind') != kind:
            continue
        if kind == 'model':
            same = live.get('model_level') == saved['model_level'] and live.get('aps_var') == saved['aps_var']
        elif kind == 'apache_system':
            same = all(live.get(field) == saved.get(field) for field in ('system_id', 'model_level', 'aps_var', 'label'))
        elif kind == 'source':
            same = _profile_enum_or_id_match(saved, live, 'source_enum', 'source_id')
        elif kind == 'energy_use':
            same = _profile_enum_or_id_match(saved, live, 'use_enum', 'use_id') and \
                   _profile_enum_or_id_match(saved, live, 'source_enum', 'source_id')
        else:
            same = all(str(live.get(field)) == str(saved.get(field)) for field in ('component_id', 'component_type', 'var_name'))
        if same:
            matches.append(key)
    if kind == 'model' and len(matches) > 1:
        matches = [key for key in matches if catalog[key].get('display_name') == saved['display_name']]
    return matches


def apply_loads_mapping_profile(profile, catalog, headers, network_components=None):
    import copy
    _mapping_profile_envelope(profile, 'hourly_loads')
    headers = _profile_headers(list(headers))
    saved_headers = _profile_headers(profile.get('columns'))
    raw_mapping = profile.get('mapping')
    results = profile.get('results')
    if not isinstance(raw_mapping, dict) or set(raw_mapping) != set(saved_headers) or not isinstance(results, dict):
        raise ExportError('Hourly mapping columns, assignments and result descriptors are inconsistent.')
    choices = {target: _profile_load_choice(raw_mapping[target], target) for target in saved_headers}
    for key, term in results.items():
        _profile_string(key, 'Saved result ID')
        _profile_load_term(term)
    referenced = {key for choice in choices.values() for key in choice['terms']}
    if referenced != set(results):
        raise ExportError('The saved result descriptors must match the selected results exactly.')
    # Native IES enum/ID objects need not support deepcopy; they must keep
    # their original runtime values for calls back into the IES API.
    working = dict(catalog)
    warnings = []
    resolved = {}
    for key, saved in results.items():
        matches = _profile_match_term(saved, catalog)
        if len(matches) == 1:
            resolved[key] = matches[0]
            if float(saved['unit_factor_to_kw']) != float(catalog[matches[0]]['unit_factor_to_kw']):
                warnings.append('Using the current IES unit conversion for ' + saved['label'] + '.')
            continue
        if matches:
            warnings.append('Saved result is ambiguous in the current simulation: ' + saved['label'])
            resolved[key] = None
            continue
        if saved['kind'] == 'hvac_component':
            available = None if network_components is None else any(
                str(component.get('id', component.get('component_id'))) == saved['component_id'] and
                str(component.get('type', component.get('component_type'))) == str(saved['component_type'])
                for component in network_components if isinstance(component, dict))
            if available is not False:
                units = {0.001: 'W', 1.0: 'kW', 1000.0: 'MW'}
                new_key, restored = _loads_new_hvac_term(saved['component_id'], saved['component_type'], saved['var_name'], units[float(saved['unit_factor_to_kw'])])
                if new_key in working:
                    # Never overwrite any live definition using a saved file.
                    resolved[key] = None
                    warnings.append('Saved HVAC result conflicts with a current definition: ' + saved['label'])
                    continue
                working[new_key] = restored
                resolved[key] = new_key
                warnings.append('Restored component result %s; its availability will be verified against the APS during export.%s' %
                                (saved['label'], ' The archived component list was unavailable.' if available is None else ''))
                continue
        resolved[key] = None
        warnings.append('Saved IES result is unavailable in the current simulation: ' + saved['label'])
    mapping = {}
    for target in headers:
        if target == 'Grid_Injection':
            mapping[target] = {'mode': 'grid', 'terms': [], 'sign': 'positive'}
            continue
        if target not in choices:
            mapping[target] = {'mode': 'unassigned', 'terms': [], 'sign': 'positive'}
            warnings.append('No saved mapping for current column %s. Select its sources before export.' % target)
            continue
        item = copy.deepcopy(choices[target])
        if any(resolved[key] is None for key in item['terms']):
            review_note = 'Saved sources are missing or ambiguous. Reselect the complete set; no partial sum was loaded.'
            mapping[target] = {'mode': 'unassigned', 'terms': [], 'sign': item['sign'],
                               'note': (item.get('note', '') + '\n' + review_note).strip()}
            warnings.append('%s needs review because at least one saved source could not be matched.' % target)
            continue
        item['terms'] = [resolved[key] for key in item['terms']]
        if len(set(item['terms'])) != len(item['terms']):
            raise ExportError('Two saved source selections resolve to the same current result in ' + target)
        mapping[target] = item
    for target in saved_headers:
        if target not in headers:
            warnings.append('Saved column %s is not in the current Excel template and was ignored.' % target)
    _validate_partial_load_mapping(mapping, working)
    return {'mapping': mapping, 'catalog': working, 'warnings': warnings}


def _show_mapping_import_result(parent, warnings):
    from tkinter import messagebox
    if warnings:
        shown = warnings[:10]
        text = 'Mapping loaded. Review the flagged rows before exporting.\n\n' + '\n'.join('- ' + message for message in shown)
        if len(warnings) > len(shown):
            text += '\n... and %s more items. Check all REQUIRED / UNASSIGNED rows.' % (len(warnings) - len(shown))
        messagebox.showwarning('Review loaded mapping', text, parent=parent)
    else:
        messagebox.showinfo('Mapping loaded', 'Mapping loaded successfully. Review the selections and continue when ready.', parent=parent)


def review_mapping(parent, uses, info, api, on_ready=None, template_targets=None,
                   target_sheet='Energy_Balance'):
    import tkinter as tk
    from tkinter import ttk, messagebox, simpledialog, filedialog
    targets = list(BASE_END_USES if template_targets is None else template_targets)
    original_template_targets = list(targets)
    added_targets = []
    win = tk.Toplevel(parent);win.title('BPVis Energy export ' + VERSION + ' — review mapping');win.geometry('900x650')
    # A transient window can inherit its parent's withdrawn state. This root
    # is hidden during file selection, so keep the review independently visible.
    if parent.winfo_viewable():
        win.transient(parent)
    ttk.Label(win,text=target_sheet + ' — MONTHLY CONSUMED ENERGY (not thermal heating/cooling demand)',font=('',11,'bold')).pack(padx=12,pady=8)
    ttk.Label(win,text='Review every used IES end use. Heat-rejection auxiliaries default to Other.\n'
              'Choose a template column, or Add new column... to create one. Meter/source splits are automatic.').pack(padx=12,pady=4)
    year_row=ttk.Frame(win);year_row.pack(fill='x',padx=12,pady=6)
    ttk.Label(year_row,text='Simulation year (metadata):').pack(side='left')
    default_year = info.get('simulation_year') or info.get('plot_year') or dt.date.today().year
    year_var=tk.StringVar(master=win,value=str(default_year));ttk.Entry(year_row,textvariable=year_var,width=8,state='readonly').pack(side='left',padx=8)
    ttk.Label(year_row,text='APS days %s–%s; %s samples/day (month lengths from APS day count)' % (info['first_day'],info['last_day'],info['results_per_day'])).pack(side='left')
    frame=ttk.Frame(win);frame.pack(fill='both',expand=True,padx=12)
    canvas=tk.Canvas(frame,highlightthickness=0);bar=ttk.Scrollbar(frame,orient='vertical',command=canvas.yview)
    canvas.configure(yscrollcommand=bar.set);bar.pack(side='right',fill='y');canvas.pack(side='left',fill='both',expand=True)
    inner=ttk.Frame(canvas);canvas.create_window((0,0),window=inner,anchor='nw')
    inner.bind('<Configure>',lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
    def choices():
        return ['UNASSIGNED'] + targets + [ADD_NEW_COLUMN]
    variables={}; comboboxes={}; previous={}
    target_lookup = {target.casefold(): target for target in targets}
    def select_target(uid):
        selected = variables[uid].get()
        if selected != ADD_NEW_COLUMN:
            previous[uid] = selected
            return
        variables[uid].set(previous[uid])
        draft = ''
        while True:
            name = simpledialog.askstring('Add energy-use column',
                'Name of the new energy-use column (for example, Refrigeration).\n'
                'The script adds _kWh automatically.', parent=win, initialvalue=draft)
            # Native/modal dialogs release their own grab when closing.
            if win.winfo_exists():
                win.grab_set()
            if name is None:
                return
            draft = name
            try:
                target = new_column_target(name, targets)
            except ExportError as exc:
                messagebox.showerror('Column name', str(exc), parent=win)
                continue
            targets.append(target); added_targets.append(target)
            for combo in comboboxes.values():
                combo.configure(values=choices())
            variables[uid].set(target); previous[uid] = target
            return
    for i,use in enumerate(uses):
        uid=id_key(use['id'])
        suggested=DEFAULT_MAPPING.get(use['enum_name'], '')
        default=target_lookup.get(suggested.casefold(), 'UNASSIGNED')
        ttk.Label(inner,text='%s [%s]' % (use.get('name',uid),uid),width=55).grid(row=i,column=0,sticky='w',pady=4)
        var=tk.StringVar(master=win,value=default);variables[uid]=var;previous[uid]=default
        combo=ttk.Combobox(inner,textvariable=var,values=choices(),state='readonly',width=28)
        combo.grid(row=i,column=1,padx=10,pady=4)
        combo.bind('<<ComboboxSelected>>',lambda event, key=uid:select_target(key))
        comboboxes[uid]=combo
    result={}
    def save_mapping():
        try:
            path = filedialog.asksaveasfilename(parent=win, title='Save monthly energy mapping',
                initialfile='BPVis_Energy_Mapping.json', defaultextension='.json',
                filetypes=[('JSON mapping', '*.json')])
            if not path:
                return
            profile = make_energy_mapping_profile(uses, targets, {key: var.get() for key, var in variables.items()})
            write_mapping_profile(path, profile)
            messagebox.showinfo('Mapping saved', 'Monthly energy mapping saved:\n' + path, parent=win)
        except Exception as exc:
            messagebox.showerror('Mapping not saved', str(exc), parent=win)
        finally:
            if win.winfo_exists():
                win.grab_set()
    def load_mapping():
        try:
            path = filedialog.askopenfilename(parent=win, title='Load monthly energy mapping',
                filetypes=[('JSON mapping', '*.json')])
            if not path:
                return
            profile = read_mapping_profile(path, 'monthly_energy')
            loaded = apply_energy_mapping_profile(profile, uses, original_template_targets)
            # Apply only after the entire profile has been checked, so an
            # invalid file cannot leave half of the current setup changed.
            targets[:] = loaded['targets']
            added_targets[:] = loaded['added_targets']
            for combo in comboboxes.values():
                combo.configure(values=choices())
            for key, var in variables.items():
                var.set(loaded['mapping'][key]); previous[key] = var.get()
            _show_mapping_import_result(win, loaded['warnings'])
        except Exception as exc:
            messagebox.showerror('Mapping not loaded', str(exc), parent=win)
        finally:
            if win.winfo_exists():
                win.grab_set()
    def finish(mode):
        if mode=='export':
            try:
                year=int(year_var.get());check_calendar(info,year)
                if any(v.get() not in targets for v in variables.values()):
                    raise ExportError('Assign every used IES category before exporting.')
            except Exception as exc:
                messagebox.showerror('Review required',str(exc),parent=win);return
            result.update(mode=mode,year=year,mapping={k:v.get() for k,v in variables.items()},
                          output_targets=list(targets),added_targets=list(added_targets))
        else:
            result.update(mode=mode,output_targets=list(targets),added_targets=list(added_targets),
                          mapping={k:v.get() for k,v in variables.items()})
        win.destroy()
    file_buttons=ttk.Frame(win);file_buttons.pack(fill='x',padx=12,pady=(6,0))
    ttk.Button(file_buttons,text='Save mapping...',command=save_mapping).pack(side='left')
    ttk.Button(file_buttons,text='Load mapping...',command=load_mapping).pack(side='left',padx=8)
    buttons=ttk.Frame(win);buttons.pack(fill='x',padx=12,pady=12)
    export_button=ttk.Button(buttons,text='Export ' + target_sheet,command=lambda:finish('export'))
    export_button.pack(side='right',padx=5)
    ttk.Button(buttons,text='Save diagnostics only',command=lambda:finish('diagnostics')).pack(side='right',padx=5)
    ttk.Button(buttons,text='Cancel',command=win.destroy).pack(side='left')
    show_window(win)
    win.grab_set()
    export_button.focus_set()
    if on_ready:
        on_ready()
    try:
        parent.wait_window(win)
    finally:
        if win.winfo_exists():
            win.destroy()
    return result


class ExportProgress:
    """Visible progress while native IES calls run on the script thread."""
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        self.window = tk.Toplevel(root)
        self.window.title('BPVis export ' + VERSION)
        self.window.geometry('620x170')
        self.window.protocol('WM_DELETE_WINDOW', lambda: None)
        ttk.Label(self.window, text='Exporting IES results to BPVis', font=('', 11, 'bold')).pack(padx=16, pady=12)
        self.text = tk.StringVar(master=self.window, value='Preparing energy extraction...')
        ttk.Label(self.window, textvariable=self.text, wraplength=580).pack(padx=16, pady=6)
        ttk.Label(self.window, text='Progress is also recorded in the audit JSON and VEScripts console.').pack(padx=16, pady=8)
        show_window(self.window)

    def update(self, message):
        self.text.set(message)
        self.window.update()

    def close(self):
        if self.window.winfo_exists():
            self.window.destroy()






"""Hourly worksheet I/O for integration into the single-file BPVis exporter.

Depends on existing ExportError/build_sheet_xml for the integrated writer;
all implementation dependencies are Python standard-library modules.
"""
import copy
import hashlib
import math
import os
from pathlib import Path
import posixpath
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from xml.parsers import expat
from zipfile import ZipFile

try:
    ExportError
except NameError:
    class ExportError(RuntimeError):
        pass

NS = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
LOADS_CALENDAR_NAMES = frozenset(('hoy', 'doy', 'day', 'month', 'weekday', 'hour', 'year'))
LOADS_REQUIRED_CALENDAR = frozenset(('hoy', 'doy', 'day', 'month', 'weekday', 'hour'))
LOADS_STANDARD_HEADERS = (
    'Heating_load', 'Cooling_load', 'HeatRejection_load', 'Ventilation_load',
    'Lighting_load', 'Equipment_load', 'HotWater_load', 'Humidification_load',
    'Dehumidification_load', 'Pumps_load', 'Other_load', 'PV_Electricity_load',
    'Electricity_load', 'Grid_Injection', 'Gas_load', 'District_Heating_load',
    'District_Cooling_load',
)


def named_sheet_path(workbook, sheet_name):
    document = ET.fromstring(workbook.read('xl/workbook.xml'))
    links = {r.attrib['Id']: r.attrib['Target'] for r in
             ET.fromstring(workbook.read('xl/_rels/workbook.xml.rels'))}
    for sheet in document.findall('{%s}sheets/{%s}sheet' % (NS, NS)):
        if sheet.attrib['name'] == sheet_name:
            target = links[sheet.attrib['{%s}id' % REL]]
            path = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/' + target)
            if not path.startswith('xl/') or '..' in path.split('/'):
                raise ExportError('Invalid worksheet path for ' + sheet_name)
            return path
    raise ExportError('Template has no exact %s worksheet.' % sheet_name)


def _loads_column_number(letters):
    number = 0
    for letter in letters:
        number = number * 26 + ord(letter) - 64
    return number


def _loads_tag_end(xml, start):
    """Index after >, respecting quotes in a start tag."""
    quote = None
    for index in range(start, len(xml)):
        byte = xml[index]
        if quote:
            if byte == quote:
                quote = None
        elif byte in (34, 39):
            quote = byte
        elif byte == 62:
            return index + 1
    raise ExportError('Unterminated XML tag in Loads_Balance.')


def _scan_loads_xml(xml, row_callback):
    """Stream worksheet rows, retaining byte positions and only one row in RAM."""
    if not isinstance(xml, bytes):
        raise ExportError('Loads_Balance XML must be bytes.')
    if xml.startswith((b'\xff\xfe', b'\xfe\xff')) or b'\x00' in xml[:100]:
        raise ExportError('Loads_Balance must use UTF-8 XML encoding.')
    parser = expat.ParserCreate(namespace_separator='|')
    stack = []
    namespace_stack = [{'xml': 'http://www.w3.org/XML/1998/namespace'}]
    pending_namespaces = {}
    state = {'row': None, 'cell': None, 'saw_root': False, 'sheet_data': 0}
    main = NS + '|'
    mc = 'http://schemas.openxmlformats.org/markup-compatibility/2006|'

    def start_namespace(prefix, uri):
        pending_namespaces[prefix or ''] = uri

    def start(name, attributes):
        namespaces = namespace_stack[-1]
        if pending_namespaces:
            namespaces = dict(namespaces)
            namespaces.update(pending_namespaces)
            pending_namespaces.clear()
        namespace_stack.append(namespaces)
        for attr in ('Ignorable', 'MustUnderstand', 'ProcessContent', 'PreserveElements', 'PreserveAttributes'):
            for value in attributes.get(mc + attr, '').split():
                prefix = value.split(':', 1)[0]
                if attr in ('ProcessContent', 'PreserveElements', 'PreserveAttributes') and ':' not in value:
                    continue
                if not namespaces.get(prefix):
                    raise ExportError('Unbound Loads_Balance compatibility namespace: ' + prefix)
        if name == mc + 'Choice':
            for prefix in attributes.get('Requires', '').split():
                if not namespaces.get(prefix):
                    raise ExportError('Unbound Loads_Balance compatibility namespace: ' + prefix)
        parent = stack[-1] if stack else None
        if not stack:
            if name != main + 'worksheet':
                raise ExportError('Unexpected Loads_Balance worksheet XML.')
            state['saw_root'] = True
        if name == main + 'sheetData' and parent == main + 'worksheet':
            state['sheet_data'] += 1
        if name in (main + 'tableParts', main + 'mergeCells') and parent == main + 'worksheet':
            raise ExportError('Loads_Balance uses a table or merged cells; use the plain hourly-grid template.')
        if name == main + 'row' and parent == main + 'sheetData':
            try:
                number = int(attributes['r'])
            except (KeyError, ValueError) as exc:
                raise ExportError('Loads_Balance row has an invalid row number.') from exc
            if number < 1:
                raise ExportError('Loads_Balance row number must be positive.')
            position = parser.CurrentByteIndex
            end = _loads_tag_end(xml, position)
            token = xml[position:end]
            state['row'] = {'number': number, 'start': position, 'open_end': end,
                            'self_closing': token.rstrip().endswith(b'/>'), 'cells': [],
                            'other_children': [], 'depth': len(stack),
                            'qname': token[1:].split(None, 1)[0].split(b'>', 1)[0].rstrip(b'/')}
        elif state['row'] is not None and len(stack) == state['row']['depth'] + 1:
            row = state['row']
            if name == main + 'c':
                ref = attributes.get('r', '')
                match = re.fullmatch(r'([A-Z]+)([1-9][0-9]*)', ref)
                if not match or int(match.group(2)) != row['number']:
                    raise ExportError('Invalid Loads_Balance cell reference: ' + ref)
                position = parser.CurrentByteIndex
                end = _loads_tag_end(xml, position)
                token = xml[position:end]
                state['cell'] = {'ref': ref, 'column': _loads_column_number(match.group(1)),
                                 'letters': match.group(1), 'start': position, 'open_end': end,
                                 'self_closing': token.rstrip().endswith(b'/>'),
                                 'qname': token[1:].split(None, 1)[0].split(b'>', 1)[0].rstrip(b'/'),
                                 'kind': attributes.get('t'), 'v': [], 'text': [],
                                 'has_formula': False}
            else:
                row['other_children'].append(parser.CurrentByteIndex)
        elif state['cell'] is not None and name == main + 'f':
            state['cell']['has_formula'] = True
        stack.append(name)

    def characters(text):
        cell = state['cell']
        if cell is not None and stack:
            if stack[-1] == main + 'v':
                cell['v'].append(text)
            elif stack[-1] == main + 't':
                cell['text'].append(text)

    def end(name):
        row, cell = state['row'], state['cell']
        if cell is not None and name == main + 'c' and len(stack) == row['depth'] + 2:
            cell['end'] = cell['open_end'] if cell['self_closing'] else _loads_tag_end(xml, parser.CurrentByteIndex)
            cell['v'] = ''.join(cell['v'])
            cell['text'] = ''.join(cell['text'])
            row['cells'].append(cell)
            state['cell'] = None
        if row is not None and name == main + 'row' and len(stack) == row['depth'] + 1:
            row['close_start'] = row['open_end'] if row['self_closing'] else parser.CurrentByteIndex
            row['end'] = row['open_end'] if row['self_closing'] else _loads_tag_end(xml, parser.CurrentByteIndex)
            columns = [c['column'] for c in row['cells']]
            if columns != sorted(set(columns)):
                raise ExportError('Loads_Balance row %s has duplicate or unordered cells.' % row['number'])
            row_callback(row)
            state['row'] = None
        stack.pop()
        namespace_stack.pop()

    def reject_doctype(*args):
        raise ExportError('Loads_Balance XML must not contain a DTD.')

    parser.StartNamespaceDeclHandler = start_namespace
    parser.StartElementHandler = start
    parser.CharacterDataHandler = characters
    parser.EndElementHandler = end
    parser.StartDoctypeDeclHandler = reject_doctype
    parser.ExternalEntityRefHandler = lambda *args: 0
    try:
        parser.Parse(xml, True)
    except expat.ExpatError as exc:
        raise ExportError('Invalid Loads_Balance XML: ' + str(exc)) from exc
    if not state['saw_root'] or state['sheet_data'] != 1:
        raise ExportError('Loads_Balance must contain one sheetData grid.')


def _loads_cell_value(cell, shared_strings):
    if cell['kind'] == 'inlineStr':
        return cell['text']
    if cell['kind'] == 's':
        try:
            index = int(cell['v'])
            if index < 0:
                raise ValueError()
            return shared_strings[index]
        except (ValueError, IndexError) as exc:
            raise ExportError('Invalid shared-string cell ' + cell['ref']) from exc
    return cell['v']


def read_loads_template(template):
    """Read and validate the existing hourly grid without generating dates."""
    with ZipFile(template) as workbook:
        sheet_path = named_sheet_path(workbook, 'Loads_Balance')
        xml = workbook.read(sheet_path)
        shared = []
        if 'xl/sharedStrings.xml' in workbook.namelist():
            root = ET.fromstring(workbook.read('xl/sharedStrings.xml'))
            shared = [''.join(t.text or '' for t in item.iter('{%s}t' % NS))
                      for item in root.findall('{%s}si' % NS)]
    headers = {}
    calendars = {}
    data_rows = []
    calendar_hash = hashlib.sha256()
    row_numbers = []
    hour_origin = [None]
    date_pairs = []

    def number(cell, name):
        try:
            value = float(_loads_cell_value(cell, shared))
        except (ValueError, TypeError) as exc:
            raise ExportError('Calendar field %s in %s must have a numeric value.' % (name, cell['ref'])) from exc
        if not math.isfinite(value) or value != int(value):
            raise ExportError('Calendar field %s in %s must be an integer.' % (name, cell['ref']))
        return int(value)

    def consume(row):
        r = row['number']
        if row_numbers and r <= row_numbers[-1]:
            raise ExportError('Loads_Balance rows must be ordered and unique.')
        row_numbers.append(r)
        by_column = {c['column']: c for c in row['cells']}
        if r == 1:
            for c in row['cells']:
                label = _loads_cell_value(c, shared)
                if not label:
                    continue
                if not isinstance(label, str) or not label.strip():
                    raise ExportError('Invalid hourly column header in ' + c['ref'])
                headers[c['column']] = label
            if not headers or len({s.casefold() for s in headers.values()}) != len(headers):
                raise ExportError('Loads_Balance headers are empty or duplicated.')
            if sorted(headers) != list(range(1, max(headers) + 1)):
                raise ExportError('Loads_Balance headers must be contiguous.')
            for col, label in headers.items():
                if label.strip().casefold() in LOADS_CALENDAR_NAMES:
                    calendars[col] = label.strip().casefold()
            if len(set(calendars.values())) != len(calendars):
                raise ExportError('Loads_Balance contains duplicate calendar field names.')
            if not LOADS_REQUIRED_CALENDAR.issubset(set(calendars.values())):
                raise ExportError('Loads_Balance must retain hoy, doy, day, month, weekday and hour columns.')
            for col in sorted(calendars):
                c = by_column[col]
                calendar_hash.update(xml[c['start']:c['end']])
            return
        if not headers:
            raise ExportError('Loads_Balance header row 1 is missing.')
        # Allow purely styled empty rows beyond the data, without treating them
        # as hours or modifying them. Any actual cell value requires a calendar.
        has_values = any(c['v'] or c['text'] or c['has_formula'] for c in row['cells'])
        if not has_values:
            return
        expected_row = len(data_rows) + 2
        if r != expected_row:
            raise ExportError('Loads_Balance hourly rows must be contiguous, starting at row 2.')
        if not set(calendars).issubset(by_column):
            raise ExportError('Loads_Balance row %s is missing existing calendar cells.' % r)
        values = {name: by_column[col] for col, name in calendars.items()}
        index = len(data_rows)
        if number(values['hoy'], 'hoy') != index + 1:
            raise ExportError('Loads_Balance hoy is not consecutive at row %s.' % r)
        if number(values['doy'], 'doy') != index // 24 + 1:
            raise ExportError('Loads_Balance doy does not follow 24 existing hourly rows per day at row %s.' % r)
        hour = number(values['hour'], 'hour')
        if hour_origin[0] is None:
            if hour not in (0, 1):
                raise ExportError('Loads_Balance hours must follow either 0–23 or 1–24.')
            hour_origin[0] = hour
        if hour != index % 24 + hour_origin[0]:
            raise ExportError('Loads_Balance hour sequence differs from the existing hourly row order at row %s.' % r)
        pair = (number(values['month'], 'month'), number(values['day'], 'day'))
        if index % 24 == 0:
            date_pairs.append(pair)
        elif pair != date_pairs[-1]:
            raise ExportError('Loads_Balance changes month/day within an existing 24-hour day at row %s.' % r)
        for col in sorted(calendars):
            c = by_column[col]
            if not _loads_cell_value(c, shared) and not c['has_formula']:
                raise ExportError('Empty calendar cell ' + c['ref'])
            calendar_hash.update(xml[c['start']:c['end']])
        data_rows.append(r)

    _scan_loads_xml(xml, consume)
    if len(data_rows) not in (8760, 8784):
        raise ExportError('Loads_Balance must already contain 8760 or 8784 hourly rows; found %s. '
                          'The exporter never adds or changes calendar rows.' % len(data_rows))
    month_lengths = [31, 29 if len(data_rows) == 8784 else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    expected_pairs = [(month, day) for month, days in enumerate(month_lengths, 1) for day in range(1, days+1)]
    if date_pairs != expected_pairs:
        raise ExportError('Existing Loads_Balance month/day fields do not match its full-year hourly row order. '
                          'Correct or select the template calendar; the exporter will not change it.')
    return {
        'sheet_path': sheet_path,
        'headers': [headers[c] for c in sorted(headers)],
        'column_by_header': {label: col for col, label in headers.items()},
        'calendar_columns': [headers[c] for c in sorted(calendars)],
        'calendar_headers': [headers[c] for c in sorted(calendars)],
        'load_columns': [label for col, label in headers.items() if col not in calendars],
        'load_headers': [label for col, label in headers.items() if col not in calendars],
        'standard_load_columns': [label for label in headers.values() if label in LOADS_STANDARD_HEADERS],
        'unknown_columns': [label for col, label in headers.items()
                            if col not in calendars and label not in LOADS_STANDARD_HEADERS],
        'data_rows': data_rows,
        'hours_count': len(data_rows),
        'hour_count': len(data_rows),
        'hour_origin': hour_origin[0],
        'calendar_sha256': calendar_hash.hexdigest(),
        'worksheet_sha256': hashlib.sha256(xml).hexdigest(),
    }


def validate_loads_hour_count(layout, expected_hours):
    if expected_hours not in (8760, 8784) or layout['hours_count'] != expected_hours:
        raise ExportError('APS contains %s hours but Loads_Balance has %s existing hourly rows. '
                          'Select a matching template; calendar rows will not be changed.'
                          % (expected_hours, layout['hours_count']))


def build_loads_sheet_xml(original, series_by_header, layout):
    """Patch only explicitly mapped numeric load cells; preserve every other byte."""
    if hashlib.sha256(original).hexdigest() != layout['worksheet_sha256']:
        raise ExportError('Loads_Balance changed after its layout was read. Reopen the template.')
    if not series_by_header:
        return original
    headers = layout['column_by_header']
    forbidden = set(layout['calendar_columns'])
    count = layout['hours_count']
    values_by_col = {}
    for header, values in series_by_header.items():
        if header not in headers:
            raise ExportError('Hourly target column is not in Loads_Balance: ' + header)
        if header in forbidden or header.strip().casefold() in LOADS_CALENDAR_NAMES:
            raise ExportError('Calendar columns cannot be overwritten: ' + header)
        if len(values) != count:
            raise ExportError('%s has %s values; expected %s existing hourly rows.' % (header, len(values), count))
        checked = []
        for value in values:
            try:
                number = float(value)
            except (ValueError, TypeError) as exc:
                raise ExportError('Non-numeric hourly value in ' + header) from exc
            if not math.isfinite(number):
                raise ExportError('Non-finite hourly value in ' + header)
            checked.append(format(number, '.15g').encode('ascii'))
        values_by_col[headers[header]] = checked
    row_to_index = {row: i for i, row in enumerate(layout['data_rows'])}
    edits = []
    touched_rows = set()

    def make_cell(cell, value):
        opening = original[cell['start']:cell['open_end']]
        # Numeric cells use the default n type. All other attributes, including
        # styles and local namespace declarations, remain byte-identical.
        opening = re.sub(rb'\s+([^\s=/>]+)\s*=\s*(?:"[^"]*"|\x27[^\x27]*\x27)',
                         lambda match: b'' if match.group(1) == b't' else match.group(0), opening)
        opening = re.sub(rb'/\s*>$', b'>', opening)
        qname = cell['qname']
        prefix = qname.rsplit(b':', 1)[0] + b':' if b':' in qname else b''
        return opening + b'<' + prefix + b'v>' + value + b'</' + prefix + b'v></' + qname + b'>'

    def consume(row):
        if row['number'] not in row_to_index:
            return
        index = row_to_index[row['number']]
        touched_rows.add(row['number'])
        by_column = {c['column']: c for c in row['cells']}
        missing = []
        for col in sorted(values_by_col):
            value = values_by_col[col][index]
            cell = by_column.get(col)
            if cell:
                edits.append((cell['start'], cell['end'], make_cell(cell, value)))
            else:
                # Insert in existing cell order without changing any row or
                # calendar cell. Reuse the row's spreadsheet namespace prefix.
                qname = row['qname']
                prefix = qname.rsplit(b':', 1)[0] + b':' if b':' in qname else b''
                letters = ''
                n = col
                while n:
                    n, digit = divmod(n-1, 26)
                    letters = chr(65+digit) + letters
                ref = (letters + str(row['number'])).encode('ascii')
                fragment = b'<' + prefix + b'c r="' + ref + b'"><' + prefix + b'v>' + value + b'</' + prefix + b'v></' + prefix + b'c>'
                later = [c['start'] for c in row['cells'] if c['column'] > col]
                position = min(later) if later else min(row['other_children'] + [row['close_start']])
                missing.append((position, col, fragment))
        # Several absent target cells can share one insertion point.
        for position in sorted({p for p, _, _ in missing}):
            pieces = [fragment for p, _, fragment in sorted(missing) if p == position]
            edits.append((position, position, b''.join(pieces)))

    _scan_loads_xml(original, consume)
    if touched_rows != set(layout['data_rows']):
        raise ExportError('Hourly row layout no longer matches Loads_Balance.')
    edits.sort(key=lambda item: (item[0], item[1]))
    pieces, cursor = [], 0
    for start, end, replacement in edits:
        if start < cursor:
            raise ExportError('Overlapping hourly worksheet edits.')
        pieces.extend((original[cursor:start], replacement))
        cursor = end
    pieces.append(original[cursor:])
    updated = b''.join(pieces)
    # Check the saved XML and its compatibility prefixes before workbook write.
    _scan_loads_xml(updated, lambda row: None)
    return updated


# Paired workbook versions. All worksheet data/calendars are copied or patched
# by the existing writers; only package metadata is changed to append a pair.


def _version_dom_bytes(document):
    validate_markup_namespaces(document)
    return document.toxml(encoding='utf-8')


def _version_children(node, namespace, local_name):
    return [child for child in node.childNodes
            if child.nodeType == child.ELEMENT_NODE
            and child.namespaceURI == namespace and child.localName == local_name]


def _version_element(document, parent, namespace, local_name):
    return document.createElementNS(namespace,
        (parent.prefix + ':' if parent.prefix else '') + local_name)


def _version_relationship_path(sheet_path):
    return posixpath.join(posixpath.dirname(sheet_path), '_rels', posixpath.basename(sheet_path) + '.rels')


def _version_check_sheet(workbook, name, sheet_path):
    """Keep unsupported owned objects out rather than duplicate object identities."""
    raw = workbook.read(sheet_path)
    for tag in ('tableParts', 'drawing', 'legacyDrawing', 'legacyDrawingHF',
                'oleObjects', 'controls', 'extLst', 'picture'):
        if re.search(rb'<(?:[A-Za-z_][\w.-]*:)?' + tag.encode('ascii') + rb'(?=[\s/>])', raw):
            raise ExportError('Cannot create a new version of %s: it contains %s. '
                              'Use the plain-grid BPVis data sheets for versioned export.' % (name, tag))
    if re.search(rb'<(?:[A-Za-z_][\w.-]*:)?sheetPr\b[^>]*\bcodeName\s*=', raw):
        raise ExportError('Cannot duplicate the worksheet code name in %s. Use a plain-grid .xlsx template.' % name)
    rel_path = _version_relationship_path(sheet_path)
    if rel_path in workbook.namelist():
        root = ET.fromstring(workbook.read(rel_path))
        seen = set()
        for rel in root:
            rid = rel.get('Id')
            if not rid or rid in seen:
                raise ExportError('Invalid or duplicate sheet relationship in ' + name)
            seen.add(rid)
            if (rel.get('TargetMode') != 'External' or
                    rel.get('Type') != REL + '/hyperlink'):
                raise ExportError('Cannot create a new version of %s: it has an unsupported sheet-owned part. '
                                  'Use plain-grid BPVis data sheets.' % name)
    return raw


def _version_plan(workbook):
    if len(workbook.namelist()) != len(set(workbook.namelist())):
        raise ExportError('The workbook contains duplicate ZIP members.')
    root = ET.fromstring(workbook.read('xl/workbook.xml'))
    sheets = root.findall('{%s}sheets/{%s}sheet' % (NS, NS))
    names = [sheet.get('name', '') for sheet in sheets]
    if len(names) != len({name.casefold() for name in names}):
        raise ExportError('The workbook contains duplicate worksheet names.')
    suffix = 2
    for name in names:
        match = re.fullmatch(r'(?:Energy_Balance|Loads_Balance)([0-9]+)', name, re.IGNORECASE)
        if match:
            suffix = max(suffix, int(match.group(1)) + 1)
    target_names = {'energy': 'Energy_Balance' + str(suffix),
                    'loads': 'Loads_Balance' + str(suffix)}
    if any(len(name) > 31 for name in target_names.values()):
        raise ExportError('The next version number would exceed Excel\'s worksheet-name limit.')
    source_names = {'energy': 'Energy_Balance', 'loads': 'Loads_Balance'}
    source_paths = {}
    for key, name in source_names.items():
        source_paths[key] = named_sheet_path(workbook, name)
        _version_check_sheet(workbook, name, source_paths[key])
    if source_paths['energy'] == source_paths['loads']:
        raise ExportError('The two BPVis data sheets cannot share one worksheet part.')
    return {'suffix': suffix, 'source_names': source_names,
            'target_names': target_names, 'source_paths': source_paths}


def plan_workbook_version(template):
    """Preview the next matched suffix without touching the workbook."""
    with ZipFile(template) as workbook:
        return _version_plan(workbook)


def _version_rewrite_reference(text, names):
    # Known simple BPVis sheet names only. Exclude tokens embedded in external
    # workbook references or longer sheet names. Local names retain other text.
    pieces = re.split(r'("(?:[^"]|"")*")', text)
    for index in range(0, len(pieces), 2):
        for old, new in names.items():
            pattern = r"(?<![A-Za-z0-9_.'\]])(?:'" + re.escape(old) + r"'|" + re.escape(old) + r')!'
            pieces[index] = re.sub(pattern, "'" + new + "'!", pieces[index], flags=re.IGNORECASE)
    return ''.join(pieces)


def _version_update_app_properties(raw, added_names):
    app_ns = 'http://schemas.openxmlformats.org/officeDocument/2006/extended-properties'
    vt = 'http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes'
    with minidom.parseString(raw) as document:
        root = document.documentElement
        headings = _version_children(root, app_ns, 'HeadingPairs')
        titles = _version_children(root, app_ns, 'TitlesOfParts')
        if not headings or not titles:
            return raw  # Optional metadata is absent in the supplied template.
        h_vectors = _version_children(headings[0], vt, 'vector')
        t_vectors = _version_children(titles[0], vt, 'vector')
        if len(h_vectors) != 1 or len(t_vectors) != 1:
            raise ExportError('Cannot safely update workbook sheet-title metadata.')
        variants = _version_children(h_vectors[0], vt, 'variant')
        title_nodes = [node for node in t_vectors[0].childNodes if node.nodeType == node.ELEMENT_NODE]
        if len(variants) % 2:
            raise ExportError('Invalid workbook heading-pair metadata.')
        position = 0
        found = False
        for index in range(0, len(variants), 2):
            label_nodes = _version_children(variants[index], vt, 'lpstr')
            count_nodes = _version_children(variants[index + 1], vt, 'i4')
            if len(label_nodes) != 1 or len(count_nodes) != 1:
                raise ExportError('Invalid workbook heading-pair metadata.')
            label = ''.join(n.data for n in label_nodes[0].childNodes if n.nodeType == n.TEXT_NODE)
            try:
                count = int(''.join(n.data for n in count_nodes[0].childNodes if n.nodeType == n.TEXT_NODE))
            except ValueError as exc:
                raise ExportError('Invalid workbook sheet-title count.') from exc
            if count < 0:
                raise ExportError('Invalid workbook sheet-title count.')
            if label.casefold() == 'worksheets':
                if found:
                    raise ExportError('Duplicate workbook worksheet-title metadata.')
                found = True
                insert_at = position + count
                if insert_at > len(title_nodes):
                    raise ExportError('Workbook sheet-title metadata is inconsistent.')
                before = title_nodes[insert_at] if insert_at < len(title_nodes) else None
                for name in added_names:
                    node = _version_element(document, t_vectors[0], vt, 'lpstr')
                    node.appendChild(document.createTextNode(name))
                    t_vectors[0].insertBefore(node, before)
                count_nodes[0].replaceChild(document.createTextNode(str(count + len(added_names))), count_nodes[0].firstChild)
            position += count
        if not found:
            return raw
        t_vectors[0].setAttribute('size', str(len(title_nodes) + len(added_names)))
        return _version_dom_bytes(document)


def _version_clone_pair(workbook, patched_sheets):
    plan = _version_plan(workbook)
    package_ns = 'http://schemas.openxmlformats.org/package/2006/relationships'
    content_ns = 'http://schemas.openxmlformats.org/package/2006/content-types'
    replacements = {}
    additions = {}
    names = workbook.namelist()
    with minidom.parseString(workbook.read('xl/workbook.xml')) as book, \
         minidom.parseString(workbook.read('xl/_rels/workbook.xml.rels')) as rels, \
         minidom.parseString(workbook.read('[Content_Types].xml')) as content:
        validate_markup_namespaces(book)
        sheet_groups = _version_children(book.documentElement, NS, 'sheets')
        if len(sheet_groups) != 1:
            raise ExportError('Workbook sheets metadata is missing or duplicated.')
        group = sheet_groups[0]
        sheets = _version_children(group, NS, 'sheet')
        sheet_ids = [int(node.getAttribute('sheetId')) for node in sheets]
        if len(sheet_ids) != len(set(sheet_ids)):
            raise ExportError('Workbook sheet IDs are duplicated.')
        next_sheet_id = max(sheet_ids, default=0) + 1
        rel_nodes = _version_children(rels.documentElement, package_ns, 'Relationship')
        rel_ids = [node.getAttribute('Id') for node in rel_nodes]
        if len(rel_ids) != len(set(rel_ids)) or '' in rel_ids:
            raise ExportError('Workbook relationship IDs are invalid or duplicated.')
        used_rel_ids = set(rel_ids)
        overrides = _version_children(content.documentElement, content_ns, 'Override')
        content_by_path = {node.getAttribute('PartName'): node.getAttribute('ContentType') for node in overrides}
        if len(content_by_path) != len(overrides):
            raise ExportError('Workbook content-type overrides are duplicated.')
        name_map = {plan['source_names'][key]: plan['target_names'][key] for key in ('energy', 'loads')}
        defined_groups = _version_children(book.documentElement, NS, 'definedNames')
        if len(defined_groups) > 1:
            raise ExportError('Workbook defined-name groups are duplicated.')
        defined_names = _version_children(defined_groups[0], NS, 'definedName') if defined_groups else []
        for offset, key in enumerate(('energy', 'loads')):
            source_name, target_name = plan['source_names'][key], plan['target_names'][key]
            source_path = plan['source_paths'][key]
            source_index = next(index for index, node in enumerate(sheets) if node.getAttribute('name') == source_name)
            source_node = sheets[source_index]
            old_rid = source_node.getAttributeNS(REL, 'id')
            source_rel = next((node for node in rel_nodes if node.getAttribute('Id') == old_rid), None)
            if source_rel is None or source_rel.getAttribute('Type') != REL + '/worksheet':
                raise ExportError('Missing or unsupported worksheet relationship for ' + source_name)
            number = 1
            while ('xl/worksheets/sheet%s.xml' % number in names or
                   'xl/worksheets/sheet%s.xml' % number in additions or
                   '/xl/worksheets/sheet%s.xml' % number in content_by_path):
                number += 1
            new_path = 'xl/worksheets/sheet%s.xml' % number
            rel_number = 1
            while 'rId' + str(rel_number) in used_rel_ids:
                rel_number += 1
            rid = 'rId' + str(rel_number)
            used_rel_ids.add(rid)
            sheet = _version_element(book, group, NS, 'sheet')
            sheet.setAttribute('name', target_name)
            sheet.setAttribute('sheetId', str(next_sheet_id + offset))
            sheet.setAttribute('state', 'visible')
            # Copy the original attribute spelling and any declaration local to
            # its sheet element, rather than assuming the prefix is always r.
            rid_attr = source_node.getAttributeNodeNS(REL, 'id')
            sheet.setAttributeNS(REL, rid_attr.name, rid)
            if rid_attr.prefix:
                sheet.setAttribute('xmlns:' + rid_attr.prefix, REL)
            group.appendChild(sheet)
            relation = _version_element(rels, rels.documentElement, package_ns, 'Relationship')
            relation.setAttribute('Type', REL + '/worksheet')
            relation.setAttribute('Target', '/' + new_path)
            relation.setAttribute('Id', rid)
            rels.documentElement.appendChild(relation)
            content_type = content_by_path.get('/' + source_path)
            if not content_type or not content_type.endswith('.worksheet+xml'):
                raise ExportError('Missing worksheet content type for ' + source_name)
            override = _version_element(content, content.documentElement, content_ns, 'Override')
            override.setAttribute('PartName', '/' + new_path)
            override.setAttribute('ContentType', content_type)
            content.documentElement.appendChild(override)
            additions[new_path] = patched_sheets.get(source_path, workbook.read(source_path))
            # Modern Excel may assign a revision UID to the worksheet root.
            # Give only that root UID a new identity; no cell bytes are touched.
            import uuid
            additions[new_path] = re.sub(
                rb'(<(?:[A-Za-z_][\w.-]*:)?worksheet\b[^>]*?\s(?:[A-Za-z_][\w.-]*:)?uid\s*=\s*)(["\x27])[^"\x27]*\2',
                lambda match: match.group(1) + match.group(2) + ('{' + str(uuid.uuid4()).upper() + '}').encode('ascii') + match.group(2),
                additions[new_path], count=1)
            old_rel_path = _version_relationship_path(source_path)
            if old_rel_path in names:
                new_rel_path = _version_relationship_path(new_path)
                if new_rel_path in names:
                    raise ExportError('New worksheet relationship path already exists.')
                additions[new_rel_path] = workbook.read(old_rel_path)
            for original in defined_names:
                if original.getAttribute('localSheetId') != str(source_index):
                    continue
                cloned = original.cloneNode(True)
                cloned.setAttribute('localSheetId', str(len(sheets) + offset))
                for node in cloned.childNodes:
                    if node.nodeType in (node.TEXT_NODE, node.CDATA_SECTION_NODE):
                        node.data = _version_rewrite_reference(node.data, name_map)
                defined_groups[0].appendChild(cloned)
        calc = _version_children(book.documentElement, NS, 'calcPr')
        if calc:
            calc[0].setAttribute('fullCalcOnLoad', '1')
            calc[0].setAttribute('forceFullCalc', '1')
        replacements['xl/workbook.xml'] = _version_dom_bytes(book)
        replacements['xl/_rels/workbook.xml.rels'] = _version_dom_bytes(rels)
        replacements['[Content_Types].xml'] = _version_dom_bytes(content)
    if 'docProps/app.xml' in names:
        updated = _version_update_app_properties(workbook.read('docProps/app.xml'),
            [plan['target_names'][key] for key in ('energy', 'loads')])
        if updated != workbook.read('docProps/app.xml'):
            replacements['docProps/app.xml'] = updated
    return replacements, additions, plan


def write_workbook(template, destination, columns=None, loads_series=None,
                   loads_layout=None, new_version=False):
    """Export into the original data sheets, or append a numbered sheet pair."""
    template, destination = Path(template), Path(destination)
    if template.resolve() == destination.resolve() or destination.exists():
        raise ExportError('Choose a NEW output filename. Existing files are never overwritten.')
    if columns is not None and len(columns) > 16383:
        raise ExportError('Too many monthly columns for Excel.')
    if columns is None and loads_series is None:
        raise ExportError('No monthly or hourly results were supplied for export.')
    if loads_series is not None and loads_layout is None:
        loads_layout = read_loads_template(template)
    tmp = None
    destination_created = False
    try:
        with ZipFile(template) as src:
            replacements, additions = {}, {}
            if columns is not None:
                member = named_sheet_path(src, 'Energy_Balance')
                replacements[member] = build_sheet_xml(src.read(member), columns)
            if loads_series is not None:
                member = named_sheet_path(src, 'Loads_Balance')
                if member != loads_layout['sheet_path']:
                    raise ExportError('Loads_Balance sheet path changed since inspection.')
                replacements[member] = build_loads_sheet_xml(src.read(member), loads_series, loads_layout)
            if new_version:
                replacements, additions, _ = _version_clone_pair(src, replacements)
            fd, tmp = tempfile.mkstemp(prefix='bpvis_', suffix='.xlsx', dir=str(destination.parent))
            os.close(fd)
            with ZipFile(tmp, 'w') as out:
                out.comment = src.comment
                for entry in src.infolist():
                    out.writestr(copy.copy(entry), replacements.get(entry.filename, src.read(entry.filename)))
                for name, data in additions.items():
                    from zipfile import ZIP_DEFLATED
                    out.writestr(name, data, compress_type=ZIP_DEFLATED)
            with ZipFile(tmp) as check:
                if src.namelist() + list(additions) != check.namelist():
                    raise ExportError('Workbook members changed unexpectedly.')
                for entry in src.infolist():
                    expected = replacements.get(entry.filename, src.read(entry.filename))
                    if check.read(entry.filename) != expected:
                        raise ExportError('Workbook preservation check failed: ' + entry.filename)
                for name, data in additions.items():
                    if check.read(name) != data:
                        raise ExportError('New worksheet preservation check failed: ' + name)
                if check.testzip() is not None:
                    raise ExportError('Output ZIP integrity check failed.')
        with open(destination, 'xb') as output:
            destination_created = True
            with open(tmp, 'rb') as source:
                shutil.copyfileobj(source, output)
    except BaseException:
        if destination_created:
            destination.unlink(missing_ok=True)
        raise
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


"""Hourly engine fragment for integration into the BPVis IESVE exporter.

Uses existing ExportError, check_calendar, vector, id_key, math and GENERATION_ENUMS.
All returned load arrays are hourly mean kW; calendar fields are never generated.
"""

HOURLY_LOAD_TARGETS = (
    'Heating_load', 'Cooling_load', 'HeatRejection_load', 'Ventilation_load',
    'Lighting_load', 'Equipment_load', 'HotWater_load', 'Humidification_load',
    'Dehumidification_load', 'Pumps_load', 'Other_load', 'PV_Electricity_load',
    'Electricity_load', 'Grid_Injection', 'Gas_load', 'District_Heating_load',
    'District_Cooling_load',
)

_ELECTRIC_USE_GROUPS = {
    'Ventilation_load': {'prm_fans_interior_central', 'prm_fans_interior_local',
                         'prm_fans_garage', 'prm_fans_exhaust', 'prm_fans_process'},
    'Lighting_load': {'prm_interior_lighting', 'prm_exterior_lighting',
                      'prm_interior_lighting_process', 'prm_interior_lighting_unregulated'},
    'Equipment_load': {'prm_receptacle_equipment', 'prm_data_center_equipment', 'prm_cooking'},
    'Pumps_load': {'prm_pumps'},
    'Other_load': {'prm_elevators_escalators', 'prm_refrigeration', 'prm_other_process',
                   'prm_motor', 'prm_transformer', 'prm_heat_rejection'},
}

# Inclusion relationships, not merely similar labels. Selecting both a total
# and its own component in one output would count that demand twice.
_MODEL_LOAD_INCLUDES = {
    'Cooling coils load': {'ApHVAC cooling coils sensible load', 'ApHVAC cooling coils latent load'},
    'Boilers load': {'Ap Sys boilers load', 'ApHVAC boilers load'},
    'ApHVAC boilers load': {"ApHVAC boilers space cond'g load", 'ApHVAC boilers DHW load'},
    'Ap Sys boilers load': {"Ap Sys boilers space cond'g load", 'Ap Sys boilers DHW load'},
    'Chillers load': {'Ap Sys chillers load', 'ApHVAC chillers load'},
    'ApHVAC hot water loop load': {"ApHVAC hot water loop space cond'g load", 'ApHVAC hot water loop DHW load'},
    'ApHVAC generic heat source load': {'ApHVAC generic heat source sp cond load', 'ApHVAC generic heat source DHW load'},
    'ApHVAC heat transfer loop heating load': {'ApHVAC heat transfer loop space htg load', 'ApHVAC heat transfer loop DHW load'},
}


def _load_unit_factor(units, unit_type):
    item = units.get(unit_type, {}).get('units_metric', {}) if isinstance(units, dict) else {}
    label = str(item.get('display_name', '')).strip()
    scales = {'W': 0.001, 'kW': 1.0, 'MW': 1000.0}
    if label not in scales:
        return None
    try:
        divisor = float(item['divisor'])
        offset = float(item.get('offset', 0))
    except (ValueError, TypeError, KeyError):
        return None
    if not math.isfinite(divisor) or divisor <= 0 or not math.isfinite(offset) or abs(offset) > 1e-12:
        return None
    return scales[label] / divisor


def _load_term_id(*parts):
    # JSON encoding keeps separators inside a variable/name unambiguous.
    import json
    return json.dumps([str(p) for p in parts], ensure_ascii=False, separators=(',', ':'))


def _source_enum_name(api, source_id):
    enum = getattr(api, 'EnergySource', None)
    if enum is not None:
        for name in dir(enum):
            if not name.startswith('_') and id_key(getattr(enum, name)) == id_key(source_id):
                return name
    # These IDs are documented by IESVE 2026. Never infer district cooling
    # from a guessed numeric ID or from a translated source display name.
    return {'1': 'elec', '2': 'nat_gas', '19': 'grid_disp_elec',
            '40': 'grid_disp_elec_pv', '41': 'district_heating'}.get(id_key(source_id), '')


def _model_load_boundary(aps_var, level):
    if level == 'e':
        return 'energy_consumption'
    if aps_var in ('Heating coils load', 'Cooling coils load', 'HVAC room units htg load',
                   'HVAC room units clg load', 'Steam humidifiers load',
                   'ApHVAC cooling coils sensible load', 'ApHVAC cooling coils latent load'):
        return 'airside_and_room_units'
    if any(word in aps_var.lower() for word in ('loop', 'boiler', 'chiller', 'heat source', 'heat pump', 'hps ', 'hwl ', 'htl ')):
        return 'plant_or_loop'
    return 'other_thermal_boundary'


def build_load_catalog(reader, api, uses, sources, variables, units, audit):
    """Discover selectable power series; no matching by fuzzy display names."""
    catalog = {}
    rejected = []
    for var in variables if isinstance(variables, list) else []:
        level = str(var.get('model_level', ''))
        if level not in ('l', 'e'):
            continue
        factor = _load_unit_factor(units, var.get('units_type'))
        if factor is None:
            rejected.append({'display_name': var.get('display_name'), 'units_type': var.get('units_type')})
            continue
        aps_var = str(var.get('aps_varname', ''))
        display = str(var.get('display_name', ''))
        if not aps_var or not display:
            continue
        key = _load_term_id('model', level, aps_var, display)
        quantity = 'thermal' if level == 'l' else 'energy'
        if level == 'e' and aps_var in ('Total electricity', 'PV generated electricity'):
            quantity = 'electricity'
        catalog[key] = {'kind': 'model', 'label': 'Building total | %s [%s]' % (display, level),
                        'aps_var': aps_var, 'display_name': display, 'model_level': level,
                        'units_type': var.get('units_type'), 'unit_factor_to_kw': factor,
                        'quantity': quantity, 'scope': 'whole_model',
                        'boundary': _model_load_boundary(aps_var, level)}
    for source in sources:
        sid = source['id']; enum_name = _source_enum_name(api, sid)
        generation = enum_name in ('grid_disp_elec', 'grid_disp_elec_pv')
        key = _load_term_id('source', id_key(sid))
        catalog[key] = {'kind': 'source', 'label': 'All uses / all meters | %s [source %s]' % (source.get('name', sid), id_key(sid)),
                        'source_id': sid, 'source_enum': enum_name, 'generation': generation,
                        'quantity': 'electricity' if enum_name in ('elec', 'grid_disp_elec', 'grid_disp_elec_pv') else 'energy',
                        'scope': 'source_total', 'boundary': 'energy_consumption', 'unit_factor_to_kw': 0.001}
        # Every used non-generation use is retained for gross-electricity
        # reconciliation, including heating/cooling and any custom/unknown use.
        if enum_name == 'elec':
            for use in uses:
                if use.get('enum_name') in GENERATION_ENUMS:
                    continue
                uid = use['id']; term_id = _load_term_id('energy_use', id_key(uid), id_key(sid))
                catalog[term_id] = {
                    'kind': 'energy_use', 'label': 'Electricity / all meters | %s [use %s]' % (use.get('name', uid), id_key(uid)),
                    'use_id': uid, 'source_id': sid, 'use_enum': use.get('enum_name', ''),
                    'source_enum': enum_name, 'generation': False, 'quantity': 'electricity',
                    'scope': 'end_use_source', 'boundary': 'energy_consumption', 'unit_factor_to_kw': 0.001,
                }
    # Source-unspecified generation avoids versions which do not support
    # querying PV/wind with a constrained energy source.
    for use in uses:
        if use.get('enum_name') not in GENERATION_ENUMS:
            continue
        uid = use['id']; key = _load_term_id('energy_use', id_key(uid), '*')
        catalog[key] = {'kind': 'energy_use', 'label': 'Generation / all meters | %s [use %s]' % (use.get('name', uid), id_key(uid)),
                        'use_id': uid, 'use_enum': use.get('enum_name'), 'generation': True,
                        'quantity': 'electricity', 'scope': 'end_use_total',
                        'boundary': 'generation', 'unit_factor_to_kw': 0.001}
    # Apache Systems and ApacheHVAC are different APIs. Only explicit v-level
    # power variables are offered here; do not fabricate HVAC component IDs.
    try:
        systems = reader.get_apache_systems()
        audit['hourly_apache_systems'] = systems
    except Exception as exc:
        systems = []
        audit['hourly_apache_system_discovery'] = {'error': str(exc)}
    for system in systems or []:
        if not isinstance(system, (list, tuple)) or len(system) < 2:
            continue
        name, system_id = system[0], system[1]
        for var in variables if isinstance(variables, list) else []:
            if var.get('model_level') != 'v':
                continue
            factor = _load_unit_factor(units, var.get('units_type'))
            if factor is None:
                continue
            aps_var = str(var.get('aps_varname', '')); display = str(var.get('display_name', ''))
            if not aps_var or not display:
                continue
            key = _load_term_id('apache_system', system_id, aps_var, display)
            catalog[key] = {'kind': 'apache_system', 'label': 'Apache System %s [%s] | %s' % (name, system_id, display),
                            'system_id': system_id, 'aps_var': aps_var, 'display_name': display, 'model_level': 'v',
                            'units_type': var.get('units_type'), 'unit_factor_to_kw': factor,
                            'quantity': 'thermal', 'scope': 'apache_system', 'boundary': 'apache_system_load'}
    audit['hourly_catalog_count'] = len(catalog)
    audit['hourly_excluded_nonpower_variables'] = rejected
    return catalog


def get_default_load_mapping(catalog, uses, sources=None, info=None, **unused):
    mapping = {name: {'mode': 'unassigned', 'terms': [], 'sign': 'positive'} for name in HOURLY_LOAD_TARGETS}
    def exact_model(aps_var, level):
        return [key for key, spec in catalog.items() if spec['kind'] == 'model'
                and spec.get('model_level') == level and spec.get('aps_var') == aps_var]
    def model_preset(target, names, level='l', sign='positive'):
        groups = [exact_model(name, level) for name in names]
        # Ambiguous aliases or missing pieces require user review. Never
        # silently export only half of a compound thermal default.
        if all(len(group) == 1 for group in groups):
            mapping[target] = {'mode': 'sum', 'terms': [group[0] for group in groups], 'sign': sign}
    # User-selected plant boundaries: all HW-loop space heating, all CW-loop
    # total load and all DHW-system demand. Exact model-level aggregate names
    # below are reviewable suggestions; they are not sums of guessed IDs.
    model_preset('Heating_load', ["ApHVAC hot water loop space cond'g load"])
    mapping['Heating_load']['note'] = ('IES names this whole-model aggregate space conditioning load. '
        'Verify HWL space-heating scope if hot-water loops also serve absorption cooling.')
    model_preset('Cooling_load', ['ApHVAC chilled water loop load'], sign='magnitude')
    model_preset('HotWater_load', ['ApHVAC DHW system DHW demand load'])
    model_preset('Humidification_load', ['Steam humidifiers load'])
    model_preset('Dehumidification_load', ['ApHVAC cooling coils latent load'], sign='magnitude')
    if info is not None and not info.get('hvac_file'):
        # A generic catalogue may list HVAC variables even when the APS was
        # simulated using Apache Systems only. Never assume those are active.
        for target in ('Heating_load', 'Cooling_load', 'HotWater_load',
                       'Humidification_load', 'Dehumidification_load'):
            mapping[target] = {'mode': 'unassigned', 'terms': [], 'sign': 'positive'}
        # Do not substitute room/airside terminal results for loop demands.
        # Apache Systems-only users must choose a matching boundary explicitly.
    model_preset('Electricity_load', ['Total electricity'], level='e')
    model_preset('PV_Electricity_load', ['PV generated electricity'], level='e', sign='magnitude')
    electrical_pairs = [(key, spec) for key, spec in catalog.items()
                        if spec['kind'] == 'energy_use' and spec.get('source_enum') == 'elec']
    electric_sources = [spec for spec in catalog.values() if spec['kind'] == 'source' and spec.get('source_enum') == 'elec']
    for target, enums in _ELECTRIC_USE_GROUPS.items():
        terms = [key for key, spec in electrical_pairs if spec.get('use_enum') in enums]
        mapping[target] = {'mode': 'sum' if terms else 'zero', 'terms': terms, 'sign': 'positive'}
    if not electric_sources:
        mapping['Electricity_load'] = {'mode': 'zero', 'terms': [], 'sign': 'positive'}
    for target, enum_name in (('Gas_load', 'nat_gas'), ('District_Heating_load', 'district_heating')):
        terms = [key for key, spec in catalog.items() if spec['kind'] == 'source' and spec.get('source_enum') == enum_name]
        mapping[target] = {'mode': 'sum' if terms else 'zero', 'terms': terms, 'sign': 'positive'}
    pv_uses = [use for use in uses if use.get('enum_name') == 'prm_elec_gen_pv']
    if not pv_uses:
        mapping['PV_Electricity_load'] = {'mode': 'zero', 'terms': [], 'sign': 'positive'}
    elif mapping['PV_Electricity_load']['mode'] == 'unassigned':
        terms = [key for key, spec in catalog.items() if spec['kind'] == 'energy_use'
                 and spec.get('use_enum') == 'prm_elec_gen_pv' and spec.get('source_id') is None]
        if len(terms) == 1:
            mapping['PV_Electricity_load'] = {'mode': 'sum', 'terms': terms, 'sign': 'magnitude'}
    mapping['Grid_Injection'] = {'mode': 'grid', 'terms': [], 'sign': 'positive'}
    # Heat rejection has no universal whole-model total; district cooling has
    # no dedicated EnergySource enum. User must select terms or explicit zero.
    return mapping


def _selected_term_overlaps(terms, catalog, target):
    if len(terms) != len(set(terms)):
        raise ExportError('A result is selected more than once for ' + target)
    for i, key in enumerate(terms):
        a = catalog[key]
        for other in terms[i+1:]:
            b = catalog[other]
            source_a = a.get('source_id'); source_b = b.get('source_id')
            if source_a is not None and source_b is not None and id_key(source_a) == id_key(source_b):
                if a['kind'] == 'source' or b['kind'] == 'source':
                    raise ExportError('Do not add a source total and its end-use breakdown in ' + target)
            names = {a.get('aps_var'), b.get('aps_var')}
            if a['kind'] == 'model' and b['kind'] == 'model' and a.get('model_level') == b.get('model_level'):
                if a.get('aps_var') == b.get('aps_var'):
                    raise ExportError('Duplicate aliases of the same IES result in ' + target)
                for parent, child in ((a.get('aps_var'), b.get('aps_var')), (b.get('aps_var'), a.get('aps_var'))):
                    if child in _MODEL_LOAD_INCLUDES.get(parent, set()):
                        raise ExportError('Do not add an aggregate and its component in %s: %s / %s' % (target, parent, child))
            if 'Total electricity' in names:
                other_spec = b if a.get('aps_var') == 'Total electricity' else a
                if other_spec.get('source_enum') == 'elec':
                    raise ExportError('Gross electricity already includes the selected electricity breakdown in ' + target)
            if 'PV generated electricity' in names:
                other_spec = b if a.get('aps_var') == 'PV generated electricity' else a
                if other_spec.get('use_enum') == 'prm_elec_gen_pv' or other_spec.get('source_enum') == 'grid_disp_elec_pv':
                    raise ExportError('PV generation has been selected twice in ' + target)
            if a['kind'] == 'energy_use' and b['kind'] == 'energy_use' and id_key(a.get('use_id')) == id_key(b.get('use_id')):
                if source_a is None or source_b is None or id_key(source_a) == id_key(source_b):
                    raise ExportError('An end-use aggregate overlaps its selected breakdown in ' + target)
            if a['kind'] == 'hvac_component' and b['kind'] == 'hvac_component':
                signature = lambda spec: (str(spec.get('component_id')), str(spec.get('component_type')), spec.get('var_name'))
                if signature(a) == signature(b):
                    raise ExportError('The same HVAC component result is selected twice in ' + target)


def _positive_load_values(values, sign, label):
    if sign not in ('positive', 'negative', 'magnitude'):
        raise ExportError('Invalid sign convention for ' + label)
    scale = max((abs(v) for v in values), default=0.0)
    tolerance_kw = max(0.0001, scale * 1e-5)
    if sign == 'negative':
        converted = [-value for value in values]
    elif sign == 'magnitude':
        if min(values, default=0.0) < -tolerance_kw and max(values, default=0.0) > tolerance_kw:
            raise ExportError('Mixed-sign result in %s. Choose an appropriate demand/output variable; absolute values would hide reversals.' % label)
        converted = [-value for value in values] if math.fsum(values) < 0 else list(values)
    else:
        converted = list(values)
    if min(converted, default=0.0) < -tolerance_kw:
        raise ExportError('Result sign does not match the selected sign convention for ' + label)
    return [max(0.0, value) for value in converted]


def _reconcile_load_kw(expected, actual, label, hours_per_sample):
    if len(expected) != len(actual):
        raise ExportError('Array-length mismatch in ' + label)
    for i, (a, b) in enumerate(zip(expected, actual)):
        if abs(a-b) > max(0.0001, abs(a)*0.0001, abs(b)*0.0001):
            raise ExportError('%s does not reconcile at sample %s (%.9g / %.9g kW).' % (label, i+1, a, b))
    expected_kwh = math.fsum(expected)*hours_per_sample
    actual_kwh = math.fsum(actual)*hours_per_sample
    if abs(expected_kwh-actual_kwh) > max(0.01, abs(expected_kwh)*0.0001):
        raise ExportError('Annual energy does not reconcile for ' + label)
    return {'native_kwh': expected_kwh, 'partition_kwh': actual_kwh, 'status': 'PASS'}


def extract_loads(reader, catalog, mapping, info, hour_count, audit, progress=print):
    count, per_day = check_calendar(info, info.get('simulation_year', 2001))
    if per_day < 24 or per_day % 24:
        raise ExportError('Hourly export needs 24 or an integer multiple of 24 results per day; no interpolation is performed.')
    samples_per_hour = per_day // 24
    expected_hours = count // samples_per_hour
    if hour_count != expected_hours:
        raise ExportError('Template has %s hourly rows but APS has %s. Calendar rows will not be inserted, deleted or shifted.' % (hour_count, expected_hours))
    cache = {}
    requested = set()
    audit['hourly_term_reads'] = []
    audit['hourly_reconciliation'] = []
    audit['hourly_warnings'] = [
        'Dehumidification can be a component of the cooling demand; these columns are not necessarily additive.',
        'Select one consistent system boundary per output. Whole-model totals already include all matching systems; do not also add their individual components.',
        'Thermal defaults use whole-model HW-loop, CW-loop and DHW-system aggregates. Review against the requested component boundaries; do not also add their individual loops/systems.',
        "The suggested heating variable is ApHVAC hot water loop space cond'g load. If hot-water loops also drive absorption cooling, verify the space-heating boundary or select explicit HW-loop space-heating component results.",
    ]
    equipment_terms = set(mapping.get('Equipment_load', {}).get('terms', []))
    other_terms = set(mapping.get('Other_load', {}).get('terms', []))
    if equipment_terms & other_terms:
        raise ExportError('The same result cannot be allocated to both Equipment_load and Other_load.')
    for target, choice in mapping.items():
        mode = choice.get('mode')
        if target == 'Grid_Injection' and mode != 'grid':
            raise ExportError('Grid_Injection must be calculated from hourly PV generation and gross electricity demand.')
        if mode == 'unassigned' or mode not in ('sum', 'zero', 'grid'):
            raise ExportError('Assign hourly column ' + target + ' or explicitly select zero/not present.')
        terms = choice.get('terms', [])
        if mode == 'sum' and not terms:
            raise ExportError('No IES result selected for ' + target)
        if mode != 'sum' and terms:
            raise ExportError('Unexpected selected results in non-sum mapping for ' + target)
        if mode == 'grid' and target != 'Grid_Injection':
            raise ExportError('Calculated grid-surplus mode is only valid for Grid_Injection.')
        for term in terms:
            if term not in catalog:
                raise ExportError('Unknown hourly result selected for ' + target)
            requested.add(term)
        _selected_term_overlaps(terms, catalog, target)
        boundaries = {catalog[term].get('boundary') for term in terms}
        if 'plant_or_loop' in boundaries and 'airside_and_room_units' in boundaries:
            audit['hourly_warnings'].append(target + ': plant/loop and airside terminal results are combined; verify that these represent distinct systems, not the same heat twice.')

    def read_term(term_id, missing_pair_ok=False):
        if term_id in cache:
            return cache[term_id]
        spec = catalog[term_id]
        kind = spec['kind']; label = spec.get('label', term_id)
        args = {'start_day': int(reader.first_day), 'end_day': int(reader.last_day)}
        try:
            if kind == 'model':
                raw = reader.get_results(spec['aps_var'], spec['display_name'], spec['model_level'], **args)
            elif kind == 'apache_system':
                raw = reader.get_apache_system_results(spec['system_id'], spec['aps_var'], spec['display_name'], spec.get('model_level', 'v'), **args)
            elif kind == 'hvac_component':
                raw = reader.get_hvac_component_results(spec['component_id'], int(spec['component_type']), spec['var_name'], **args)
            elif kind in ('energy_use', 'source'):
                args.update(type='e', add_subs=-1)
                if kind == 'energy_use':
                    args['use_id'] = spec['use_id']
                if spec.get('source_id') is not None:
                    args['source_id'] = spec['source_id']
                raw = reader.get_energy_results(**args)
            else:
                raise ExportError('Unsupported hourly result kind: ' + kind)
            values = vector(raw, count, label)
        except Exception as exc:
            raise ExportError('IES hourly read failed for %s: %s' % (label, exc)) from exc
        missing = values is None
        if missing:
            if not missing_pair_ok:
                raise ExportError('Selected IES result is unavailable: ' + label)
            values = [0.0]*count
        factor = float(spec.get('unit_factor_to_kw', 0))
        if not math.isfinite(factor) or factor <= 0:
            raise ExportError('Missing or invalid power-unit conversion for ' + label)
        converted = [value*factor for value in values]
        cache[term_id] = converted
        audit['hourly_term_reads'].append({'term_id': term_id, 'label': label, 'kind': kind,
            'specification': dict(spec), 'unit_factor_to_kw': factor,
            'missing_pair_treated_as_zero_pending_reconciliation': missing,
            'sample_count': count, 'minimum_raw_kw': min(converted), 'maximum_raw_kw': max(converted)})
        return converted

    # An end use can be marked used while consuming only gas. Treat its missing
    # electricity intersection as zero only after the entire non-generation
    # electrical end-use partition matches independently read gross demand.
    selected_electric_pairs = [term for term in requested if catalog[term]['kind'] == 'energy_use'
                               and catalog[term].get('source_enum') == 'elec']
    if selected_electric_pairs:
        gross_terms = [key for key, spec in catalog.items() if spec['kind'] == 'model'
                       and spec.get('model_level') == 'e' and spec.get('aps_var') == 'Total electricity']
        if len(gross_terms) != 1:
            raise ExportError('Electricity end-use export requires an unambiguous native Total electricity result for reconciliation.')
        progress('Verifying all electricity end uses against gross electricity demand...')
        gross = _positive_load_values(read_term(gross_terms[0]), 'positive', 'Total electricity')
        partition = [0.0]*count
        pair_keys = [key for key, spec in catalog.items() if spec['kind'] == 'energy_use'
                     and spec.get('source_enum') == 'elec' and not spec.get('generation')]
        for key in pair_keys:
            values = _positive_load_values(read_term(key, missing_pair_ok=True), 'positive', catalog[key].get('label', key))
            partition = [a+b for a,b in zip(partition, values)]
        report = _reconcile_load_kw(gross, partition, 'Electricity end-use partition', 24.0/per_day)
        report.update(check='all non-generation electrical end uses versus native Total electricity',
                      pair_count=len(pair_keys), missing_pairs=[entry['term_id'] for entry in audit['hourly_term_reads']
                      if entry['missing_pair_treated_as_zero_pending_reconciliation']])
        audit['hourly_reconciliation'].append(report)
        source_terms = [key for key, spec in catalog.items() if spec['kind'] == 'source' and spec.get('source_enum') == 'elec']
        if len(source_terms) == 1:
            source_total = _positive_load_values(read_term(source_terms[0]), 'positive', 'Electricity source aggregate')
            report = _reconcile_load_kw(gross, source_total, 'Electricity source total', 24.0/per_day)
            report['check'] = 'electricity source aggregate versus native Total electricity'
            audit['hourly_reconciliation'].append(report)

    output = {}
    for target, choice in mapping.items():
        mode = choice['mode']
        if mode == 'grid':
            continue
        progress('Reading hourly ' + target + '...')
        if mode == 'zero':
            output[target] = [0.0]*hour_count
            continue
        total = [0.0]*count
        for term in choice['terms']:
            values = _positive_load_values(read_term(term), choice.get('sign', 'positive'), catalog[term].get('label', term))
            total = [a+b for a,b in zip(total, values)]
        output[target] = [math.fsum(total[i:i+samples_per_hour])/samples_per_hour
                          for i in range(0, count, samples_per_hour)]
    if any(choice['mode'] == 'grid' for choice in mapping.values()):
        if 'PV_Electricity_load' not in output or 'Electricity_load' not in output:
            raise ExportError('Grid_Injection requires exported PV_Electricity_load and Electricity_load.')
        output['Grid_Injection'] = [max(pv-load, 0.0) for pv,load in zip(output['PV_Electricity_load'], output['Electricity_load'])]
    audit['hourly_export'] = {
        'output_unit': 'hourly mean kW', 'rows': hour_count, 'samples_per_hour': samples_per_hour,
        'hours_per_native_sample': 24.0/per_day,
        'calendar_policy': 'Existing Excel calendar cells unchanged; APS samples placed in chronological row order without date/week/day shifts.',
        'grid_injection_definition': 'max(hourly mean PV generation kW - hourly mean gross electricity demand kW, 0); hourly net surplus, calculated after averaging.',
        'mapping': mapping,
        'series': {target: {'peak_kw': max(values), 'annual_kwh': math.fsum(values),
                            'minimum_kw': min(values), 'mode': mapping[target]['mode']}
                   for target,values in output.items()},
        'unique_native_terms_read': len(cache),
    }
    return output


"""Tk helpers for the BPVis hourly loads exporter; integrated into one script.

Host supplies ExportError, VERSION and show_window. No IES or workbook mutations
occur here. Input dictionaries are copied so closing a dialog changes nothing.
"""

THERMAL_LOAD_COLUMNS = {
    'Heating_load', 'Cooling_load', 'HeatRejection_load', 'HotWater_load',
    'Humidification_load', 'Dehumidification_load',
}
ELECTRICAL_LOAD_COLUMNS = {
    'Ventilation_load', 'Lighting_load', 'Equipment_load', 'Pumps_load',
    'Other_load', 'PV_Electricity_load', 'Electricity_load',
}


def _loads_target_allows_term(target, term):
    thermal = term.get('quantity') == 'thermal' or term.get('kind') == 'hvac_component'
    if target in THERMAL_LOAD_COLUMNS:
        return thermal
    if target in ELECTRICAL_LOAD_COLUMNS:
        return term.get('quantity') == 'electricity'
    if target in {
        'Gas_load', 'District_Heating_load', 'District_Cooling_load'
    }:
        return not thermal
    return True


def _loads_validate_mapping(mapping, headers, catalog):
    for target in headers:
        item = mapping.get(target, {})
        mode = item.get('mode')
        if target == 'Grid_Injection':
            if mode != 'grid':
                raise ExportError('Grid_Injection must use the calculated hourly net surplus.')
            continue
        if mode == 'zero':
            continue
        if mode != 'sum' or not item.get('terms'):
            raise ExportError('Select results for %s, or explicitly choose Not present — write zero.' % target)
        terms = item['terms']
        if len(set(terms)) != len(terms):
            raise ExportError('A result is selected more than once for ' + target)
        if item.get('sign') not in ('positive', 'negative', 'magnitude'):
            raise ExportError('Select a sign convention for ' + target)
        for key in terms:
            if key not in catalog:
                raise ExportError('A selected result is unavailable for ' + target)
            if not _loads_target_allows_term(target, catalog[key]):
                raise ExportError('The selected result has the wrong quantity for ' + target)
        overlap_check = globals().get('_selected_term_overlaps')
        if callable(overlap_check):
            overlap_check(terms, catalog, target)
    equipment = mapping.get('Equipment_load', {})
    other = mapping.get('Other_load', {})
    if equipment.get('mode') == other.get('mode') == 'sum':
        duplicate = set(equipment.get('terms', [])) & set(other.get('terms', []))
        if duplicate:
            labels = [catalog.get(key, {}).get('label', str(key)) for key in sorted(duplicate)]
            raise ExportError('Equipment_load and Other_load must not contain the same result: ' +
                              '; '.join(labels) + '. Allocate each result to one of these columns.')


def _loads_new_hvac_term(component_id, component_type, var_name, unit):
    import hashlib
    import json
    component_id = str(component_id).strip()
    var_name = str(var_name).strip()
    if not component_id or not var_name:
        raise ExportError('Enter the component ID and the exact IES result variable name.')
    if any(ord(c) < 32 for c in component_id + var_name):
        raise ExportError('Component IDs and variable names cannot contain control characters.')
    try:
        component_type = int(component_type)
    except (TypeError, ValueError) as exc:
        raise ExportError('The component type must be an IES integer type ID.') from exc
    if component_type < 0:
        raise ExportError('The component type cannot be negative.')
    factors = {'W': 0.001, 'kW': 1.0, 'MW': 1000.0}
    if unit not in factors:
        raise ExportError('Select the raw result unit: W, kW or MW.')
    identity = json.dumps([component_id, component_type, var_name], ensure_ascii=False)
    key = 'hvac:' + hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]
    term = {
        'kind': 'hvac_component', 'component_id': component_id,
        'component_type': component_type, 'var_name': var_name,
        'unit_factor_to_kw': factors[unit], 'quantity': 'thermal',
        'scope': 'HVAC component', 'raw_unit': unit,
        'label': 'HVAC component %s (type %s): %s [%s raw]' %
                 (component_id, component_type, var_name, unit),
    }
    return key, term


def _loads_make_window(parent, title, geometry):
    import tkinter as tk
    win = tk.Toplevel(parent)
    win.title(title)
    win.geometry(geometry)
    if parent.winfo_viewable():
        win.transient(parent)
    return win


def _loads_wait(parent, win):
    show_window(win)
    win.grab_set()
    try:
        parent.wait_window(win)
    finally:
        if win.winfo_exists():
            win.destroy()
        if parent.winfo_exists() and parent.winfo_viewable():
            parent.grab_set()


def choose_export_scope(root):
    import tkinter as tk
    from tkinter import ttk
    win = _loads_make_window(root, 'BPVis export ' + VERSION, '650x535')
    result = {}
    ttk.Label(win, text='Choose where to import the simulation results', font=('', 12, 'bold')).pack(
        anchor='w', padx=20, pady=(18, 8))
    sheet_mode = tk.StringVar(master=win, value='existing')
    ttk.Radiobutton(win, text='Update existing Energy_Balance / Loads_Balance sheets',
        variable=sheet_mode, value='existing').pack(anchor='w', padx=24, pady=5)
    ttk.Radiobutton(win, text='Create a new project version (duplicate both sheets)',
        variable=sheet_mode, value='new').pack(anchor='w', padx=24, pady=5)
    ttk.Label(win, text='New versions use Energy_Balance2 and Loads_Balance2, then 3, 4, ... '
              'if a numbered version already exists. The original sheets are preserved.',
              wraplength=595).pack(anchor='w', padx=24, pady=(3, 10))
    ttk.Label(win, text='Select the worksheets to export', font=('', 12, 'bold')).pack(
        anchor='w', padx=20, pady=(8, 10))
    selected = tk.StringVar(master=win, value='both')
    for value, text in (
        ('both', 'Monthly Energy_Balance and hourly Loads_Balance'),
        ('loads', 'Hourly Loads_Balance only'),
        ('energy', 'Monthly Energy_Balance only'),
    ):
        ttk.Radiobutton(win, text=text, variable=selected, value=value).pack(
            anchor='w', padx=24, pady=5)
    ttk.Label(win, text='A new output file is always created. Calendar fields remain unchanged. '
              'For a new version, a sheet excluded above is copied with its existing values.',
              wraplength=595).pack(anchor='w', padx=20, pady=12)
    footer = ttk.Frame(win); footer.pack(side='bottom', fill='x', padx=20, pady=14)
    def finish():
        value = selected.get()
        result.update(energy=value in ('both', 'energy'), loads=value in ('both', 'loads'),
                      new_version=sheet_mode.get() == 'new')
        win.destroy()
    ttk.Button(footer, text='Continue', command=finish).pack(side='right')
    ttk.Button(footer, text='Cancel', command=win.destroy).pack(side='left')
    _loads_wait(root, win)
    return result or None


def _loads_component_dialog(parent, network_components):
    import tkinter as tk
    from tkinter import ttk, messagebox
    win = _loads_make_window(parent, 'Add HVAC component result', '820x440')
    answer = {}
    ttk.Label(win, text='Select an actual thermal result from one HVAC component.',
              font=('', 11, 'bold')).pack(anchor='w', padx=16, pady=(14, 8))
    ttk.Label(win, text='Use the exact IES result variable name and its raw unit. '
              'The script verifies that the selected result exists before writing Excel. '
              'Do not combine a loop total with the devices already included in that loop.',
              wraplength=775).pack(anchor='w', padx=16, pady=6)
    body = ttk.Frame(win); body.pack(fill='x', padx=16, pady=8)
    component_id = tk.StringVar(master=win, value='')
    component_type = tk.StringVar(master=win, value='')
    var_name = tk.StringVar(master=win, value='')
    unit = tk.StringVar(master=win, value='W')
    candidates = {}
    for item in network_components or []:
        if not isinstance(item, dict):
            continue
        cid = item.get('id', item.get('component_id'))
        ctype = item.get('type', item.get('component_type', item.get('aps_component_type')))
        if cid is None or ctype is None:
            continue
        name = str(item.get('name', item.get('reference', cid)))
        label = '%s | ID %s | type %s' % (name, cid, ctype)
        candidates[label] = (str(cid), str(ctype))
    manual = 'Enter component ID and type manually'
    choice = tk.StringVar(master=win, value=manual)
    ttk.Label(body, text='Component').grid(row=0, column=0, sticky='w', pady=6)
    selector = ttk.Combobox(body, textvariable=choice, state='readonly', width=75,
                            values=[manual] + sorted(candidates))
    selector.grid(row=0, column=1, columnspan=3, sticky='ew', padx=10, pady=6)
    def selected(event=None):
        item = candidates.get(choice.get())
        if item:
            component_id.set(item[0]); component_type.set(item[1])
    selector.bind('<<ComboboxSelected>>', selected)
    ttk.Label(body, text='Component ID').grid(row=1, column=0, sticky='w', pady=6)
    ttk.Entry(body, textvariable=component_id, width=32).grid(row=1, column=1, sticky='w', padx=10)
    ttk.Label(body, text='IES type ID').grid(row=1, column=2, sticky='w')
    ttk.Entry(body, textvariable=component_type, width=10).grid(row=1, column=3, sticky='w', padx=10)
    ttk.Label(body, text='Exact result name').grid(row=2, column=0, sticky='w', pady=6)
    ttk.Entry(body, textvariable=var_name, width=65).grid(row=2, column=1, columnspan=3, sticky='ew', padx=10)
    ttk.Label(body, text='Raw result unit').grid(row=3, column=0, sticky='w', pady=6)
    ttk.Combobox(body, textvariable=unit, values=['W', 'kW', 'MW'], state='readonly', width=10).grid(
        row=3, column=1, sticky='w', padx=10)
    body.columnconfigure(1, weight=1)
    footer = ttk.Frame(win); footer.pack(side='bottom', fill='x', padx=16, pady=14)
    def finish():
        try:
            key, term = _loads_new_hvac_term(component_id.get(), component_type.get(), var_name.get(), unit.get())
        except ExportError as exc:
            messagebox.showerror('Component result', str(exc), parent=win); return
        answer.update(key=key, term=term)
        win.destroy()
    ttk.Button(footer, text='Add result', command=finish).pack(side='right')
    ttk.Button(footer, text='Cancel', command=win.destroy).pack(side='left')
    _loads_wait(parent, win)
    return answer or None


def _loads_pick_sources(parent, target, catalog, current, network_components=None):
    import copy
    import tkinter as tk
    from tkinter import ttk, messagebox
    local_catalog = copy.deepcopy(catalog)
    selected = set(current.get('terms', []))
    original_order = list(current.get('terms', []))
    allowed = lambda key: _loads_target_allows_term(target, local_catalog[key])
    selected = {key for key in selected if key in local_catalog and allowed(key)}
    win = _loads_make_window(parent, target + ' — IES results', '1000x700')
    result = {}
    ttk.Label(win, text=target, font=('', 12, 'bold')).pack(anchor='w', padx=16, pady=(12, 5))
    explanation = ('Select thermal loads at one physical boundary. Building totals already sum '
                   'their matching systems; do not add the same plant, loop or coil loads again.'
                   if target in THERMAL_LOAD_COLUMNS else
                   'Select consumption or generation results. Source totals already include their '
                   'meters; do not add the same end-use or meter results a second time.')
    ttk.Label(win, text=explanation, wraplength=950).pack(anchor='w', padx=16, pady=5)
    if target == 'HeatRejection_load':
        ttk.Label(win, text='Use rejected thermal heat. Fan/pump electricity and pre-cooling-only '
                  'rejection are not a whole-system heat-rejection total.', wraplength=950).pack(
                      anchor='w', padx=16, pady=5)
    footer = ttk.Frame(win); footer.pack(side='bottom', fill='x', padx=16, pady=14)
    options = ttk.Frame(win); options.pack(side='bottom', fill='x', padx=16, pady=8)
    sign_labels = {
        'Positive values': 'positive',
        'Negative values — reverse sign': 'negative',
        'Consistent positive or negative — output magnitude': 'magnitude',
    }
    current_sign = current.get('sign', 'positive')
    sign_label = tk.StringVar(master=win, value=next(
        (label for label, key in sign_labels.items() if key == current_sign), 'Positive values'))
    ttk.Label(options, text='IES result sign').pack(side='left')
    ttk.Combobox(options, textvariable=sign_label, values=list(sign_labels), state='readonly', width=58).pack(
        side='left', padx=12)
    search_row = ttk.Frame(win); search_row.pack(fill='x', padx=16, pady=7)
    ttk.Label(search_row, text='Search').pack(side='left')
    query = tk.StringVar(master=win, value='')
    ttk.Entry(search_row, textvariable=query).pack(side='left', fill='x', expand=True, padx=10)
    count = tk.StringVar(master=win, value='')
    ttk.Label(win, textvariable=count).pack(anchor='w', padx=16, pady=3)
    list_frame = ttk.Frame(win); list_frame.pack(fill='both', expand=True, padx=16, pady=5)
    scrollbar = ttk.Scrollbar(list_frame, orient='vertical'); scrollbar.pack(side='right', fill='y')
    horizontal = ttk.Scrollbar(list_frame, orient='horizontal'); horizontal.pack(side='bottom', fill='x')
    box = tk.Listbox(list_frame, selectmode='multiple', exportselection=False,
                     yscrollcommand=scrollbar.set, xscrollcommand=horizontal.set)
    box.pack(side='left', fill='both', expand=True)
    scrollbar.configure(command=box.yview); horizontal.configure(command=box.xview)
    visible = []
    drawing = [False]
    def sync_selection(event=None):
        if drawing[0]:
            return
        selected.difference_update(visible)
        selected.update(visible[int(index)] for index in box.curselection())
        count.set('%s selected results. Selections stay selected when the search filter changes.' % len(selected))
    def redraw(*args):
        sync_selection()
        drawing[0] = True
        try:
            text = query.get().casefold().strip()
            visible[:] = sorted((key for key in local_catalog if allowed(key) and
                                 text in local_catalog[key].get('label', key).casefold()),
                                key=lambda key: local_catalog[key].get('label', key).casefold())
            box.delete(0, 'end')
            for index, key in enumerate(visible):
                box.insert('end', local_catalog[key].get('label', key))
                if key in selected:
                    box.selection_set(index)
        finally:
            drawing[0] = False
        count.set('%s selected results. Selections stay selected when the search filter changes.' % len(selected))
    query.trace_add('write', redraw)
    box.bind('<<ListboxSelect>>', sync_selection)
    def add_component():
        sync_selection()
        answer = _loads_component_dialog(win, network_components)
        if not answer:
            return
        key, term = answer['key'], answer['term']
        if key in local_catalog and local_catalog[key].get('unit_factor_to_kw') != term['unit_factor_to_kw']:
            messagebox.showerror('Component result', 'This result is already listed with a different raw unit.', parent=win)
            return
        local_catalog[key] = term
        # Reset/redraw first, then select: redraw synchronizes the old list.
        query.set('')
        redraw()
        selected.add(key)
        if key in visible:
            box.selection_set(visible.index(key))
        sync_selection()
    if target in THERMAL_LOAD_COLUMNS:
        ttk.Button(search_row, text='Add HVAC component result...', command=add_component).pack(side='right')
    def finish(mode):
        sync_selection()
        if mode == 'zero':
            item = {'mode': 'zero', 'terms': [], 'sign': 'positive', 'explicit_zero': True}
        else:
            ordered = [key for key in original_order if key in selected]
            ordered += sorted(selected - set(ordered))
            item = {'mode': 'sum', 'terms': ordered, 'sign': sign_labels[sign_label.get()]}
            try:
                _loads_validate_mapping({target: item}, [target], local_catalog)
            except ExportError as exc:
                messagebox.showerror('Select IES results', str(exc), parent=win); return
        result.update(item=item, catalog=local_catalog)
        win.destroy()
    ttk.Button(footer, text='Use selected results', command=lambda: finish('sum')).pack(side='right')
    ttk.Button(footer, text='Not present — write zero', command=lambda: finish('zero')).pack(side='right', padx=10)
    ttk.Button(footer, text='Cancel', command=win.destroy).pack(side='left')
    redraw()
    _loads_wait(parent, win)
    return result or None


def review_load_mapping(root, catalog, defaults, layout, on_ready=None, network_components=None,
                        target_sheet='Loads_Balance'):
    import copy
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    working_catalog = copy.deepcopy(catalog)
    headers = list(layout['load_headers'])
    mapping = {target: copy.deepcopy(defaults.get(target, {
        'mode': 'unassigned', 'terms': [], 'sign': 'positive'})) for target in headers}
    if 'Grid_Injection' in mapping:
        mapping['Grid_Injection'] = {'mode': 'grid', 'terms': [], 'sign': 'positive'}
    win = _loads_make_window(root, 'BPVis hourly loads ' + VERSION + ' — review mapping', '1120x760')
    result = {}
    ttk.Label(win, text='Hourly ' + target_sheet + ' — review IES results', font=('', 12, 'bold')).pack(
        anchor='w', padx=16, pady=(14, 6))
    ttk.Label(win, text='Values are hourly mean kW. Existing Excel hour, day, weekday, month and year cells '
              'are preserved. Suggested heating uses space-heating loads from all hot-water loops; '
              'cooling uses the total load of all chilled-water loops; hot water uses demand from all DHW systems. '
              'Check these automatic allocations and edit any result as needed.',
              wraplength=1070).pack(anchor='w', padx=16, pady=5)
    ttk.Label(win, text='Select non-overlapping results. Building aggregates already include multiple matching '
              'systems. Equipment suggests receptacles, data centres and cooking; Other suggests refrigeration, '
              'elevators and other uses. Electricity consumption remains separate from thermal loads.',
              wraplength=1070).pack(anchor='w', padx=16, pady=5)
    footer = ttk.Frame(win); footer.pack(side='bottom', fill='x', padx=16, pady=14)
    frame = ttk.Frame(win); frame.pack(fill='both', expand=True, padx=16, pady=7)
    canvas = tk.Canvas(frame, highlightthickness=0)
    bar = ttk.Scrollbar(frame, orient='vertical', command=canvas.yview)
    canvas.configure(yscrollcommand=bar.set); bar.pack(side='right', fill='y')
    canvas.pack(side='left', fill='both', expand=True)
    inner = ttk.Frame(canvas)
    window_id = canvas.create_window((0, 0), window=inner, anchor='nw')
    inner.bind('<Configure>', lambda event: canvas.configure(scrollregion=canvas.bbox('all')))
    canvas.bind('<Configure>', lambda event: canvas.itemconfigure(window_id, width=event.width))
    labels = {}
    def description(target):
        item = mapping[target]
        if item.get('mode') == 'grid':
            return 'Calculated: max(hourly PV electricity − hourly gross electricity demand, 0)'
        if item.get('mode') == 'zero':
            return 'No applicable consumption/load — write zero'
        if item.get('mode') != 'sum' or not item.get('terms'):
            return 'REQUIRED: select the result(s), or explicitly choose Not present — write zero'
        terms = [working_catalog.get(key, {}).get('label', 'Unavailable result: ' + str(key))
                 for key in item['terms']]
        summary = ' + '.join(terms)
        if item.get('note'):
            summary += '\n' + item['note']
        return summary
    def edit(target):
        nonlocal working_catalog
        answer = _loads_pick_sources(win, target, working_catalog, mapping[target], network_components)
        if answer:
            working_catalog = answer['catalog']
            mapping[target] = answer['item']
            labels[target].configure(text=description(target))
    for index, target in enumerate(headers):
        ttk.Label(inner, text=target, width=27).grid(row=index, column=0, sticky='nw', pady=8, padx=(0, 10))
        label = ttk.Label(inner, text=description(target), wraplength=680, justify='left')
        label.grid(row=index, column=1, sticky='ew', pady=8, padx=8); labels[target] = label
        if target != 'Grid_Injection':
            ttk.Button(inner, text='Edit...', command=lambda name=target: edit(name)).grid(
                row=index, column=2, sticky='ne', pady=8, padx=4)
    inner.columnconfigure(1, weight=1)
    def save_mapping():
        try:
            path = filedialog.asksaveasfilename(parent=win, title='Save hourly loads mapping',
                initialfile='BPVis_Loads_Mapping.json', defaultextension='.json',
                filetypes=[('JSON mapping', '*.json')])
            if not path:
                return
            profile = make_loads_mapping_profile(mapping, working_catalog, headers)
            write_mapping_profile(path, profile)
            messagebox.showinfo('Mapping saved', 'Hourly loads mapping saved:\n' + path, parent=win)
        except Exception as exc:
            messagebox.showerror('Mapping not saved', str(exc), parent=win)
        finally:
            if win.winfo_exists():
                win.grab_set()
    def load_mapping():
        nonlocal working_catalog, mapping
        try:
            path = filedialog.askopenfilename(parent=win, title='Load hourly loads mapping',
                filetypes=[('JSON mapping', '*.json')])
            if not path:
                return
            profile = read_mapping_profile(path, 'hourly_loads')
            loaded = apply_loads_mapping_profile(profile, working_catalog, headers,
                                                 network_components=network_components)
            working_catalog = loaded['catalog']
            mapping = loaded['mapping']
            for target in headers:
                labels[target].configure(text=description(target))
            _show_mapping_import_result(win, loaded['warnings'])
        except Exception as exc:
            messagebox.showerror('Mapping not loaded', str(exc), parent=win)
        finally:
            if win.winfo_exists():
                win.grab_set()
    def finish(mode):
        if mode == 'export':
            try:
                _loads_validate_mapping(mapping, headers, working_catalog)
                if 'Grid_Injection' in headers and not {'PV_Electricity_load', 'Electricity_load'} <= set(headers):
                    raise ExportError('Grid_Injection requires PV_Electricity_load and Electricity_load in the template.')
            except ExportError as exc:
                messagebox.showerror('Review required', str(exc), parent=win); return
        result.update(mode=mode, mapping=mapping, catalog=working_catalog)
        win.destroy()
    ttk.Button(footer, text='Continue export', command=lambda: finish('export')).pack(side='right')
    ttk.Button(footer, text='Save diagnostics only', command=lambda: finish('diagnostics')).pack(side='right', padx=10)
    ttk.Button(footer, text='Cancel', command=win.destroy).pack(side='left')
    ttk.Button(footer, text='Save mapping...', command=save_mapping).pack(side='left', padx=(12, 0))
    ttk.Button(footer, text='Load mapping...', command=load_mapping).pack(side='left', padx=8)
    show_window(win); win.grab_set()
    if on_ready:
        on_ready()
    try:
        root.wait_window(win)
    finally:
        if win.winfo_exists():
            win.destroy()
    return result or None


"""Read-only discovery of components in the network archived with selected APS.

Result-name presets come from IES's downloadable working-with-hvac.zip example:
https://www.iesve.com/discoveries/view/30518/ve-scripts-working-with-apachehvac
https://www.iesve.com/support/faq/zip/working-with-hvac.zip
Network/component relations: IESVE 2026 HVACNetwork, HVACChilledWaterLoop,
HVACCondenserWaterLoop, HVACDHWSystem and HVACAbstractComponent documentation.
"""


def discover_hvac_components(api, aps_path, info, audit):
    """Return descriptors; never inspect, set, save, or simulate the live model.

    Loading a network and reading documented properties is the workflow in the
    official IES result-reading example. Missing archives or failing properties
    are reported in the audit; there is no fallback to the current project.
    Presets are alternatives at different physical boundaries, not instructions
    to add coil, plant, and loop results together.
    """
    import ntpath
    import os
    from pathlib import Path

    hvac_name = str(info.get('hvac_file') or '').strip()
    result = []
    discovery = {'aps_hvac_file': hvac_name, 'status': 'NO_ARCHIVED_NETWORK',
                 'components': [], 'paths_checked': [], 'warnings': []}
    audit['hvac_discovery'] = discovery
    if not hvac_name:
        discovery['warnings'].append('The APS does not identify an archived HVAC network.')
        return result
    loader = getattr(getattr(api, 'HVACNetwork', None), 'load_network', None)
    if not callable(loader):
        discovery.update(status='API_UNAVAILABLE')
        discovery['warnings'].append('This IESVE API does not expose HVACNetwork.load_network.')
        return result

    # Windows is the IESVE host; normalize separators to allow portable tests.
    referenced = Path(hvac_name.replace('\\', os.sep).replace('/', os.sep))
    aps_parent = Path(aps_path).resolve().parent
    if referenced.is_absolute() or ntpath.isabs(hvac_name):
        candidates = [referenced]
    else:
        candidates = [aps_parent.parent / referenced, aps_parent / referenced,
                      aps_parent / referenced.name]
    unique_paths = []
    for candidate in candidates:
        if str(candidate) not in unique_paths:
            unique_paths.append(str(candidate))
    discovery['paths_checked'] = unique_paths
    existing_paths = []
    for candidate in unique_paths:
        try:
            if Path(candidate).is_file():
                existing_paths.append(candidate)
        except OSError as exc:
            discovery['warnings'].append('Cannot inspect archived network path: ' + str(exc))
    if not existing_paths:
        discovery['warnings'].append('The network referenced by this APS was not found; component results remain available for manual mapping.')
        return result
    path = existing_paths[0]
    discovery['network_path'] = path
    if len(existing_paths) > 1:
        discovery['alternate_existing_paths'] = existing_paths[1:]
        discovery['warnings'].append('Several referenced-network locations exist; used the first recorded path. Confirm it belongs to this simulation.')

    # Exact component variable names and native W units verified in IES sample.
    # No unverified presets for dry fluid coolers or condenser-water loops.
    presets = {
        2: ('Sensible load', 'Heating coil sensible load', 'heating'),
        27: ('Sensible load', 'Heating coil sensible load', 'heating'),
        3: ('Total load', 'Cooling coil total load', 'cooling'),
        25: ('Total load', 'Cooling coil total load', 'cooling'),
        13: ('Load', 'Boiler thermal output', 'heating'),
        21: ('Load', 'Boiler thermal output', 'heating'),
        14: ('Load', 'Chiller cooling load', 'cooling'),
        20: ('Load', 'Chiller cooling load', 'cooling'),
        23: ('Load', 'Chiller cooling load', 'cooling'),
        39: ('Heat rejection', 'Cooling tower heat rejection', 'heat_rejection'),
    }
    roots = ('components', 'hot_water_loops', 'chilled_water_loops',
             'dhw_systems', 'generic_heat_sources', 'vrf_systems')
    # Waterside equipment may not be included in network.components. These
    # relationships are documented by the 2026 API or official IES example.
    children_many = ('boilers', 'chillers', 'cooling_towers', 'dry_fluid_coolers',
                     'equipment', 'branches', 'secondary_loops')
    children_one = ('heat_source', 'cooling_source', 'condenser_water_loop',
                    'cooling_tower', 'dry_fluid_cooler', 'heat_rejection_device',
                    'pre_cooling_loop', 'thermal_storage_loop', 'demand_loop',
                    'supply_loop', 'solar_water_heater')
    property_errors = []

    def read_attr(obj, name, default=None):
        try:
            return getattr(obj, name, default)
        except Exception as exc:
            if len(property_errors) < 30:
                property_errors.append({'property': name, 'error': str(exc)})
            return default

    def as_objects(value):
        if value is None or isinstance(value, (str, bytes)):
            return []
        if isinstance(value, dict):
            return list(value.values())
        try:
            return list(value)
        except TypeError:
            return []

    try:
        network = loader(path)
        if network is None:
            raise ValueError('HVACNetwork.load_network returned no network.')
        queue = []
        for attr in roots:
            queue.extend(as_objects(read_attr(network, attr)))
        seen_objects = set()
        seen_components = set()
        while queue:
            component = queue.pop(0)
            if component is None or isinstance(component, (str, bytes, int, float, bool)):
                continue
            object_key = id(component)
            if object_key in seen_objects:
                continue
            seen_objects.add(object_key)
            cid = read_attr(component, 'id')
            ctype = read_attr(component, 'aps_component_type')
            try:
                component_key = (str(cid), int(ctype)) if cid is not None else None
            except (TypeError, ValueError):
                component_key = None
            if component_key is not None:
                # Native APIs can expose fresh Python wrappers for one object;
                # deduplicating native identity also bounds traversal cycles.
                if component_key in seen_components:
                    continue
                seen_components.add(component_key)
                cid, ctype = component_key
                if cid:
                    layer = read_attr(component, 'multiplex_layer_number')
                    try:
                        layer = int(layer) if layer is not None else None
                    except (TypeError, ValueError):
                        layer = str(layer)
                    descriptor = {'id': cid, 'type': ctype,
                        'name': str(read_attr(component, 'reference') or cid),
                        'system_id': str(read_attr(component, 'system_id') or ''),
                        'multiplex_id': str(read_attr(component, 'multiplex_id') or ''),
                        'multiplex_layer_number': layer, 'standard_results': []}
                    if ctype in presets:
                        var_name, label, role = presets[ctype]
                        descriptor['standard_results'].append({
                            'var_name': var_name, 'unit_factor_to_kw': 0.001,
                            'label': label, 'quantity': 'thermal', 'role': role})
                    result.append(descriptor)
            for attr in children_many:
                queue.extend(as_objects(read_attr(component, attr)))
            for attr in children_one:
                child = read_attr(component, attr)
                if child is not None:
                    queue.append(child)
        result.sort(key=lambda c: (c['name'].casefold(), c['type'], c['id']))
        tower_ids = [c['id'] for c in result if c['type'] == 39]
        # These can contain other rejection routes. Their presence is a review
        # prompt, not a claim that their loads are additional to tower loads.
        possible_other_types = {14, 18, 19, 20, 23, 24, 30, 31, 33, 34, 38, 40, 41, 42, 43, 49}
        other_types = sorted({c['type'] for c in result if c['type'] in possible_other_types})
        discovery.update(status='READ', components=result,
            cooling_tower_ids=tower_ids,
            other_potential_heat_rejection_component_types=other_types,
            heat_rejection_review_required=True,
            heat_rejection_note=(
                'Cooling tower results cover the discovered towers only. Review dry fluid coolers, '
                'direct air-cooled equipment, heat pumps and other rejection paths. Do not add a '
                'condenser-loop total to its tower output; that can count the same heat twice.'
                if tower_ids else
                'No cooling towers were discovered. Heat rejection is unassigned; select verified '
                'component results manually, or explicitly select zero only if that load is absent.'))
        if property_errors:
            discovery['property_errors'] = property_errors
            discovery['warnings'].append('Some HVAC properties could not be read; component discovery may be incomplete.')
    except Exception as exc:
        # No live-model lookup, fallback network, model mutation or silent zero.
        discovery.update(status='DISCOVERY_FAILED', error=str(exc), components=result)
    return result



def add_hvac_catalog_results(catalog, components):
    """Expose only component result names verified in the official IES sample."""
    tower_terms = []
    for component in components:
        for preset in component.get('standard_results', []):
            key, term = _loads_new_hvac_term(component['id'], component['type'], preset['var_name'], 'W')
            term.update(unit_factor_to_kw=preset['unit_factor_to_kw'], role=preset['role'],
                label='HVAC %s [%s, type %s] | %s' % (component['name'], component['id'], component['type'], preset['label']),
                boundary='heat_rejection_device' if preset['role']=='heat_rejection' else 'component_load')
            catalog[key] = term
            if preset['role'] == 'heat_rejection':
                tower_terms.append(key)
    return tower_terms



def main():
    try:
        import iesve
    except ImportError:
        print('Run this file inside IESVE: Tools > VEScripts. The iesve module is supplied by IESVE.')
        return
    import tkinter as tk
    from tkinter import filedialog, messagebox
    root = tk.Tk(); root.withdraw()
    audit = {'exporter_version': VERSION, 'status': 'STARTED'}
    audit_path = None
    progress_window = None
    def stage(status, message):
        audit.update(status=status, progress_message=message,
                     updated_at=dt.datetime.now().isoformat(timespec='seconds'))
        print('[BPVis] ' + message, flush=True)
        if audit_path:
            write_audit(audit_path, audit)
        if progress_window:
            progress_window.update(message)
    try:
        scope = choose_export_scope(root)
        if not scope:
            return
        changed_sheets = [name for key, name in [('energy', 'Energy_Balance'), ('loads', 'Loads_Balance')] if scope[key]]
        audit['scope'] = changed_sheets
        new_version = bool(scope.get('new_version', False))
        audit['worksheet_mode'] = 'new_version' if new_version else 'update_existing'
        aps = filedialog.askopenfilename(parent=root, title='Select IESVE simulation results', filetypes=[('IES results', '*.aps')])
        if not aps:
            return
        template = filedialog.askopenfilename(parent=root, title='Select BPVis template or existing workbook', filetypes=[('Excel workbook', '*.xlsx')])
        if not template:
            return
        version_plan = plan_workbook_version(template) if new_version else None
        if version_plan:
            changed_sheets = [version_plan['target_names'][key] for key in ('energy', 'loads') if scope[key]]
            audit['new_version'] = version_plan
            audit['scope'] = changed_sheets
        destination = filedialog.asksaveasfilename(parent=root, title='Save NEW BPVis workbook', defaultextension='.xlsx',
            initialfile='BPVis_IESVE_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S') + '.xlsx', filetypes=[('Excel workbook', '*.xlsx')])
        if not destination:
            return
        if Path(destination).exists() or Path(destination).resolve() == Path(template).resolve():
            raise ExportError('Output must be a new filename, not the template or an existing workbook.')
        candidate_audit_path = str(Path(destination).with_suffix('.audit.json'))
        if Path(candidate_audit_path).exists():
            raise ExportError('An audit file with that name exists. Choose a new output filename.')
        audit_path = candidate_audit_path
        audit.update(aps_file=aps, template_file=template, output_file=destination,
                     created_at=dt.datetime.now().isoformat(timespec='seconds'))
        stage('READING_TEMPLATE', 'Checking the selected Excel sheets and existing calendar rows...')
        template_targets = read_energy_template_columns(template) if scope['energy'] else None
        layout = read_loads_template(template) if scope['loads'] else None
        if template_targets is not None:
            audit['template_energy_columns'] = [target + '_kWh' for target in template_targets]
        if layout is not None:
            audit['loads_template'] = {key: layout[key] for key in
                ('hour_count', 'load_headers', 'calendar_headers', 'calendar_sha256') if key in layout}
        audit['calendar_policy'] = 'Preserve calendar cells exactly. Map simulation day/hour in existing row order. No date, year or weekday recalculation or shift.'
        stage('OPENING_RESULTS', 'Opening IES results...')
        try:
            audit['iesve_version'] = iesve.VEProject.get_current_project().get_version()
        except Exception as exc:
            audit['iesve_version'] = 'Unavailable: ' + str(exc)
        columns = None
        loads_series = None
        with iesve.ResultsReader.open(aps) as reader:
            stage('READING_METADATA', 'Reading energy uses, sources and available result variables...')
            uses, sources, meters, info = inspect_results(reader, iesve)
            audit.update(uses=uses, sources=sources, meters=meters, results_metadata=info)
            for field, method in [('variables', 'get_variables'), ('units', 'get_units')]:
                try:
                    audit[field] = getattr(reader, method)()
                except Exception as exc:
                    audit[field] = {'error': str(exc)}
            year = int(info.get('simulation_year') or info.get('plot_year') or dt.date.today().year)
            check_calendar(info, year)
            if layout is not None:
                hourly_count = int(info['last_day']) * 24
                if layout['hour_count'] != hourly_count:
                    raise ExportError('APS has %s hourly periods but Loads_Balance has %s rows. Use a matching template; calendar rows will not be added, removed or shifted.'
                                      % (hourly_count, layout['hour_count']))
                if int(info['results_per_day']) < 24 or int(info['results_per_day']) % 24:
                    raise ExportError('Hourly export requires 24 or an integer multiple of 24 APS samples per day. No interpolation is performed.')
            energy_choice = None
            if scope['energy']:
                stage('OPENING_ENERGY_MAPPING', 'Opening the monthly energy mapping review...')
                energy_choice = review_mapping(root, uses, info, iesve,
                    on_ready=lambda: stage('WAITING_FOR_ENERGY_MAPPING', 'Review monthly energy targets and continue.'),
                    template_targets=template_targets,
                    target_sheet=version_plan['target_names']['energy'] if version_plan else 'Energy_Balance')
                if not energy_choice:
                    stage('CANCELLED', 'Export cancelled; no workbook created.'); return
                audit['mapping'] = energy_choice['mapping']
                audit['added_energy_columns'] = [target + '_kWh' for target in energy_choice['added_targets']]
                audit['requested_energy_columns'] = [target + '_kWh' for target in energy_choice['output_targets']]
                if energy_choice['mode'] == 'diagnostics':
                    stage('DIAGNOSTICS_ONLY', 'Diagnostics saved; no workbook created.')
                    messagebox.showinfo('Diagnostics saved', audit_path, parent=root); return
            loads_choice = None
            if scope['loads']:
                stage('DISCOVERING_LOAD_SOURCES', 'Preparing hourly result choices and archived HVAC component list...')
                components = discover_hvac_components(iesve, aps, info, audit)
                catalog = build_load_catalog(reader, iesve, uses, sources, audit['variables'], audit['units'], audit)
                tower_terms = add_hvac_catalog_results(catalog, components)
                defaults = get_default_load_mapping(catalog, uses, sources, info=info)
                if tower_terms and audit.get('hvac_discovery', {}).get('status') == 'READ':
                    defaults['HeatRejection_load'] = {'mode': 'sum', 'terms': tower_terms,
                        'sign': 'magnitude', 'note': 'Sum of discovered cooling towers only. Check for additional dry coolers or direct equipment heat rejection.'}
                loads_choice = review_load_mapping(root, catalog, defaults, layout,
                    on_ready=lambda: stage('WAITING_FOR_LOAD_MAPPING', 'Review hourly sources; resolve every column, then export.'),
                    network_components=components if audit.get('hvac_discovery', {}).get('status') == 'READ' else None,
                    target_sheet=version_plan['target_names']['loads'] if version_plan else 'Loads_Balance')
                if not loads_choice:
                    stage('CANCELLED', 'Export cancelled; no workbook created.'); return
                audit['loads_mapping'] = loads_choice['mapping']
                audit['loads_catalog'] = loads_choice['catalog']
                if loads_choice['mode'] == 'diagnostics':
                    stage('DIAGNOSTICS_ONLY', 'Hourly source diagnostics saved; no workbook created.')
                    messagebox.showinfo('Diagnostics saved', audit_path, parent=root); return
                _loads_validate_mapping(loads_choice['mapping'], layout['load_headers'], loads_choice['catalog'])
            progress_window = ExportProgress(root)
            if energy_choice is not None:
                columns = extract_energy(reader, iesve, uses, sources, meters, info,
                    energy_choice['mapping'], energy_choice['year'], audit,
                    progress=lambda message: stage('EXTRACTING_ENERGY', message),
                    output_targets=energy_choice['output_targets'])
            if loads_choice is not None:
                loads_series = extract_loads(reader, loads_choice['catalog'], loads_choice['mapping'], info,
                    layout['hour_count'], audit, progress=lambda message: stage('EXTRACTING_LOADS', message))
        stage('WRITING_WORKBOOK', 'Writing selected sheets and checking calendar and workbook preservation...')
        write_workbook(template, destination, columns, loads_series=loads_series, loads_layout=layout,
                       new_version=new_version)
        audit['output_sha256'] = hashlib.sha256(Path(destination).read_bytes()).hexdigest()
        stage('EXPORTED_PENDING_VISTAPRO_VALIDATION', 'Excel workbook created successfully.')
        progress_window.close(); progress_window = None
        messagebox.showinfo('BPVis export completed',
            'Created:\n' + destination + '\n\nUpdated: ' + ', '.join(changed_sheets) +
            ('\nNew version sheets: ' + ', '.join(version_plan['target_names'].values()) +
             '\nOriginal Energy_Balance and Loads_Balance sheets are preserved.' if version_plan else '') +
            '\nAll calendar fields and other worksheets are preserved.\n' +
            ('Hourly loads are mean kW; grid injection is hourly net PV surplus.\n' if scope['loads'] else '') +
            'Review totals and peaks against VistaPro on the first run.\n\nAudit:\n' + audit_path, parent=root)
    except Exception as exc:
        audit.update(failed_stage=audit.get('status'), status='FAILED', error=str(exc), traceback=traceback.format_exc())
        if audit_path:
            try:
                write_audit(audit_path, audit)
            except Exception:
                pass
        traceback.print_exc()
        if progress_window:
            progress_window.close(); progress_window = None
        messagebox.showerror('Export stopped', str(exc) + '\n\nThe original workbook has not changed.\n'
            + ('The audit JSON records the stage and selected sources:\n' + audit_path if audit_path else ''), parent=root)
    finally:
        if progress_window:
            progress_window.close()
        root.destroy()


if __name__ == '__main__':
    main()