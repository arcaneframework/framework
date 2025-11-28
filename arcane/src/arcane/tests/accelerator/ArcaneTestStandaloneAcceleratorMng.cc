// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NumArray.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/IMemoryRessourceMng.h"
#include "arcane/utils/MemoryUtils.h"

#include "arcane/launcher/ArcaneLauncher.h"

#include "arcane/accelerator/core/IAcceleratorMng.h"
#include "arcane/accelerator/core/RunQueue.h"
#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/DeviceMemoryInfo.h"
#include "arccore/common/accelerator/internal/RunQueueImpl.h"

#include "arcane/accelerator/NumArrayViews.h"
#include "arcane/accelerator/RunCommandLoop.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{

void _printAllocator(IMemoryRessourceMng* mrm, eMemoryRessource r)
{
  IMemoryAllocator* allocator = mrm->getAllocator(r, false);
  std::cout << "Allocator name=" << r << " v=" << (allocator != nullptr) << "\n";
}

void _printAvailableAllocators()
{
  IMemoryRessourceMng* mrm = platform::getDataMemoryRessourceMng();
  _printAllocator(mrm, eMemoryRessource::Host);
  _printAllocator(mrm, eMemoryRessource::Device);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void _testSum(IAcceleratorMng* acc_mng)
{
  // Test la somme de deux tableaux 'a' et 'b' dans un tableau 'c'.

  int nb_value = 10000;
  NumArray<Int64, MDDim1> a(nb_value);
  NumArray<Int64, MDDim1> b(nb_value);
  NumArray<Int64, MDDim1> c(nb_value);
  for (int i = 0; i < nb_value; ++i) {
    a(i) = i + 2;
    b(i) = i + 3;
  }

  {
    auto command = makeCommand(acc_mng->queue());
    auto in_a = viewIn(command, a);
    auto in_b = viewIn(command, b);
    auto out_c = viewOut(command, c);
    command << RUNCOMMAND_LOOP1(iter, nb_value)
    {
      auto [i] = iter();
      out_c(i) = in_a(i) + in_b(i);
    };
  }

  Int64 total = 0.0;
  for (int i = 0; i < nb_value; ++i)
    total += c(i);
  std::cout << "TOTAL=" << total << "\n";
  Int64 expected_total = 100040000;
  if (total != expected_total)
    ARCANE_FATAL("Bad value for sum={0} (expected={1})", total, expected_total);

  Accelerator::DeviceMemoryInfo dmi = acc_mng->runner().deviceMemoryInfo();
  std::cout << "DeviceMemoryInfo: free_mem=" << dmi.freeMemory()
            << " total=" << dmi.totalMemory() << "\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void _testBinOp(IAcceleratorMng* acc_mng)
{
  // Test des opérateurs binaires pour les views

  int nb_value = 10000;
  NumArray<Int64, MDDim1> a(nb_value);
  NumArray<Int64, MDDim1> b(nb_value);
  for (int i = 0; i < nb_value; ++i) {
    a(i) = 1;
    b(i) = 2;
  }

  // *=
  {
    auto command = makeCommand(acc_mng->defaultQueue());
    auto in_out_a = viewInOut(command, a);
    command << RUNCOMMAND_LOOP1(iter, nb_value)
    {
      auto [i] = iter();
      in_out_a(i) *= 2.;
    };
  }
  Int64 total = 0.0;
  for (int i = 0; i < nb_value; ++i)
    total += a(i);
  std::cout << "TOTAL=" << total << "\n";
  Int64 expected_total = nb_value * 2;
  if (total != expected_total)
    ARCANE_FATAL("Bad value for operator*= {0} (expected={1})", total, expected_total);

  // +=
  {
    auto command = makeCommand(acc_mng->queue());
    auto out_a = viewOut(command, a);
    auto in_b = viewIn(command, b);
    command << RUNCOMMAND_LOOP1(iter, nb_value)
    {
      auto [i] = iter();
      out_a(i) += in_b(i);
    };
  }
  total = 0.0;
  for (int i = 0; i < nb_value; ++i)
    total += a(i);
  std::cout << "TOTAL=" << total << "\n";
  expected_total = nb_value * 4;
  if (total != expected_total)
    ARCANE_FATAL("Bad value for operator+= {0} (expected={1})", total, expected_total);

  // -= et /=
  {
    auto command = makeCommand(acc_mng->queue());
    auto in_out_a = viewInOut(command, a);
    auto in_b = viewIn(command, b);
    command << RUNCOMMAND_LOOP1(iter, nb_value)
    {
      auto [i] = iter();
      in_out_a(i) -= in_b(i);
      in_out_a(i) /= 2;
    };
  }
  total = 0.0;
  for (int i = 0; i < nb_value; ++i)
    total += a(i);
  std::cout << "TOTAL=" << total << "\n";
  expected_total = nb_value;
  if (total != expected_total)
    ARCANE_FATAL("Bad value {0} (expected={1})", total, expected_total);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueue*
_testEmptyKernelHelper(IAcceleratorMng* acc_mng, bool use_async, bool use_concurrent, bool use_no_launch_command)
{
  // Lance un kernel vide pour évaluer le coup du lancement.
  int nb_iteration = 10000;

  auto queue = makeQueue(acc_mng->defaultRunner());
  RunQueue* q2 = new RunQueue(queue);
  if (use_async)
    queue.setAsync(true);
  if (use_concurrent)
    queue.setConcurrentCommandCreation(use_concurrent);
  Int64 xbegin = platform::getRealTimeNS();
  for (int i = 0; i < nb_iteration; ++i) {
    auto command = makeCommand(queue);
    if (!use_no_launch_command)
      command << RUNCOMMAND_SINGLE(){};
  }
  Int64 xend = platform::getRealTimeNS();
  queue.barrier();
  Int64 xend2 = platform::getRealTimeNS();
  std::cout << "Time "
            << (use_async ? "ASYNC " : "SYNC  ")
            << (use_concurrent ? " CONCURRENT " : "            ")
            << (use_no_launch_command ? " NOLAUNCH " : "   LAUNCH ");

  std::cout << "Time1 (us) = " << std::setw(6) << (xend - xbegin) / 1000 << " Time2=" << std::setw(6) << (xend2 - xbegin) / 1000;
  std::cout << "  Time1/Iter (ns) = " << std::setw(6) << (xend - xbegin) / nb_iteration << " Time2=" << std::setw(6) << (xend2 - xbegin) / nb_iteration;
  std::cout << "\n";
  queue._internalImpl()->dumpStats(std::cout);
  std::cout << "\n";
  return q2;
}

void _testEmptyKernel(IAcceleratorMng* acc_mng)
{
  // On garde les références sur les files créées pour ne pas les
  // réutiliser. Cela permet d'avoir des mesures fiables sur les temps.
  Runner runner = acc_mng->runner();
  UniqueArray<RunQueue*> queues;
  queues.reserve(8);
  queues.add(_testEmptyKernelHelper(acc_mng, false, false, false));
  queues.add(_testEmptyKernelHelper(acc_mng, false, false, true));
  queues.add(_testEmptyKernelHelper(acc_mng, true, false, false));
  queues.add(_testEmptyKernelHelper(acc_mng, true, false, true));
  if (!isAcceleratorPolicy(runner.executionPolicy())) {
    queues.add(_testEmptyKernelHelper(acc_mng, false, true, false));
    queues.add(_testEmptyKernelHelper(acc_mng, false, true, true));
    queues.add(_testEmptyKernelHelper(acc_mng, true, true, false));
    queues.add(_testEmptyKernelHelper(acc_mng, true, true, true));
  }
  for (RunQueue* q : queues)
    delete q;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int _testStandaloneLauncher(const CommandLineArguments& cmd_line_args,
                            const String& method_name)
{
  ARCANE_UNUSED(method_name);
  ArcaneLauncher::init(cmd_line_args);
  StandaloneAcceleratorMng launcher{ ArcaneLauncher::createStandaloneAcceleratorMng() };
  IAcceleratorMng* acc_mng = launcher.acceleratorMng();
  RunQueue* default_queue = acc_mng->defaultQueue();
  IMemoryRessourceMng* mrm = platform::getDataMemoryRessourceMng();
  IMemoryAllocator* alloc0 = MemoryUtils::getDeviceOrHostAllocator();
  IMemoryAllocator* wanted_alloc = mrm->getAllocator(eMemoryRessource::Host);
  if (default_queue->isAcceleratorPolicy())
    wanted_alloc = mrm->getAllocator(eMemoryRessource::Device);
  if (alloc0 != wanted_alloc)
    ARCANE_FATAL("Bad allocator");

  _printAvailableAllocators();
  if (method_name == "TestSum")
    _testSum(acc_mng);
  else if (method_name == "TestBinOp")
    _testBinOp(acc_mng);
  else if (method_name == "TestEmptyKernel")
    _testEmptyKernel(acc_mng);
  else
    ARCANE_FATAL("Unknown method to test");
  return 0;
}
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_EXPORT int
arcaneTestStandaloneAcceleratorLauncher(const CommandLineArguments& cmd_line_args,
                                        const String& method_name)
{
  int r =0;
  try{
    r = _testStandaloneLauncher(cmd_line_args,method_name);
  }
  catch(const Exception& ex){
    std::cerr << "EXCEPTION: " << ex << "\n";
    throw;
  }
  return r;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
