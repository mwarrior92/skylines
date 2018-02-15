from EasIEST.mms import collector, dispatcher, mdo
from EasIEST import cdo
import json

platform = "ripe_atlas"
label = 'example_dns_meas'
my_mdo = mdo.dns.DNS(label, query_domains=['google.com', 'amazon.com'])
loc1 = cdo.TargetLocation()
loc1.set_countries(["JP"])

loc2 = cdo.TargetLocation()
loc2.set_countries(["US"])

tg1 = cdo.TargetClientGroup(loc1, target_quantity=2)
cg1 = tg1.get_ClientGroup(platform)
tg2 = cdo.TargetClientGroup(loc2, target_quantity=2)
cg2 = tg2.get_ClientGroup(platform)

cg = cdo.ClientGroup.merge(cg1, cg2)

print cg

d = dispatcher.Dispatcher(my_mdo, platform, cg)
my_mro = d.dispatch(file_path="example_meas_result.json")
print my_mro

my_mro.set('file_path', "example_meas_result.json")

c = collector.SpinningCollector(my_mro)

collector.wait_on_collectors([c])

