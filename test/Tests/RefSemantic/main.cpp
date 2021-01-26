#include <gtest/gtest.h>

#include <Tests/Environment.h>

#include <alien/data/Universe.h>

int
main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  Environment::initialize(argc, argv);

  Alien::setTraceMng(Environment::traceMng());
  Alien::setVerbosityLevel(Alien::Verbosity::Debug);

  int ret = RUN_ALL_TESTS();

  Environment::finalize();

  return ret;
}
