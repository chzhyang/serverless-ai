# [OK] k exec wrk -- wrk -s /tmp/wrk-cloudevent/post.lua -c1 -t1 -d1s --latency http://tf-function-event-perf-warm.default.svc.cluster.local
# https://www.cnblogs.com/zhangtu/p/14350625.html

# curl -H "Content-Type: application/cloudevents+json" -X POST -d "@/tmp/wrk-cloudevent/event.json" http://tf-function-event-perf-warm.default.svc.cluster.local
curl -H "Content-Type: application/cloudevents+json" \
-X POST -d "@/tmp/wrk-cloudevent/event.json" \
-o /dev/null \
-s \
-w 'time_namelookup: %{time_namelookup}\ntime_connect: %{time_connect}\ntime_starttransfer: %{time_starttransfer}\ntime_total: %{time_total}\n' \
http://tf-function-event-perf-warm.default.svc.cluster.local
