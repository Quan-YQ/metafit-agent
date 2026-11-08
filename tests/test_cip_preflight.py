"""Check CIP format interpretation and error handling without a dataset download."""
from io import BytesIO
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
try:
    import cip_preflight as cip
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import cip_preflight as cip

def csv_zip(data):
    buf=BytesIO()
    with zipfile.ZipFile(buf,'w') as z: z.writestr('sensor.csv',data)
    return zipfile.ZipFile(BytesIO(buf.getvalue()))

def test_sensor_detects_gaps_and_allows_counter_wrap():
    data=('// Update Rate: 60.0Hz\nPacketCounter;Acc_X;Acc_Y;Acc_Z;Quat_q0;Quat_q1;Quat_q2;Quat_q3\n'
          '65535;0;0;1;1;0;0;0\n0;0;0;1;1;0;0;0\n2;0;0;1;1;0;0;0\n')
    with csv_zip(data) as z:
        result, indices=cip.sensor_data(z,'sensor.csv')
    assert indices == [65535,0,2]
    assert result['counter_discontinuities']==1
    assert result['declared_rate']=='60.0Hz'
    assert result['quaternion_norm_range']==[1.0,1.0]

def test_sensor_rejects_nonfinite_values_and_missing_columns():
    data=('PacketCounter;Acc_X;Acc_Y;Acc_Z;Quat_q0;Quat_q1;Quat_q2;Quat_q3\n'
          '1;nan;0;1;1;0;0;0\n')
    with csv_zip(data) as z, pytest.raises(ValueError,match='nonfinite'):
        cip.sensor_data(z,'sensor.csv')
    with csv_zip('PacketCounter;Acc_X\n1;2\n') as z, pytest.raises(ValueError,match='missing'):
        cip.sensor_data(z,'sensor.csv')

def test_reference_reads_head_position_and_detects_frame_gap():
    workbook=BytesIO()
    with zipfile.ZipFile(workbook,'w') as b:
        b.writestr('xl/workbook.xml','<workbook xmlns="'+cip.NS['s']+'" xmlns:r="'+cip.NS['r']+'"><sheets><sheet name="Segment Position" r:id="rId1"/></sheets></workbook>')
        b.writestr('xl/_rels/workbook.xml.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>')
        b.writestr('xl/worksheets/sheet1.xml','<worksheet xmlns="'+cip.NS['s']+'"><sheetData><row><c r="A1" t="inlineStr"><is><t>Frame</t></is></c><c r="B1" t="inlineStr"><is><t>Head z</t></is></c><c r="C1" t="inlineStr"><is><t>Pelvis z</t></is></c></row><row><c r="A2"><v>0</v></c><c r="B2"><v>1.5</v></c></row><row><c r="A3"><v>2</v></c><c r="B3"><v>0.8</v></c></row></sheetData></worksheet>')
    outer=BytesIO()
    with zipfile.ZipFile(outer,'w') as z: z.writestr('sample.xlsx',workbook.getvalue())
    with zipfile.ZipFile(BytesIO(outer.getvalue())) as z: result=cip.position_sheet(z,'sample.xlsx')
    assert result['rows']==2
    assert result['nonsequential_frame_steps']==1
    assert result['head_z_min_max_raw_units']==[0.8,1.5]
