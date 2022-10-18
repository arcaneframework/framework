#include <linux/perf_event.h>
#include <linux/hw_breakpoint.h>
#include <sys/types.h>
#include <unistd.h>
#include <syscall.h>
#include <sys/ioctl.h>

#include <string.h>
#include <iostream>

long
do_perf_event_open(struct perf_event_attr* attr,pid_t pid,int cpu,int group_fd,unsigned long flags)
{
  return syscall(__NR_perf_event_open,attr,pid,cpu,group_fd,flags);
}

int
main(int argc,char *argv[])
{
  struct perf_event_attr attr { .type = PERF_TYPE_HARDWARE, .config = PERF_COUNT_HW_CPU_CYCLES };

  attr.size = sizeof(struct perf_event_attr);
  attr.exclude_kernel = 1;
  attr.exclude_hv = 1;
  attr.disabled = 1;

  pid_t pid = ::getpid();
  int cpu = -1;
  int group_fd = -1;
  unsigned long flags = 0;
  int r = do_perf_event_open(&attr,pid,cpu,group_fd,flags);
  if (r==(-1))
    std::cout << "ERROR x=" << attr.config << " error" << strerror(errno) << "\n";
  std::cout << "R=" << r << "\n";
  return r;
}
