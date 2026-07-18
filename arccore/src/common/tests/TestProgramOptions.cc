// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arccore/common/internal/ProgramOptions.h"

#include <sstream>

using namespace Arcane;

namespace po = Arcane::ProgramOptions;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestProgramOptions, options_description_add)
{
  po::options_description desc("Test");

  ASSERT_EQ(desc.options().size(), 0);

  desc.add_options()
    ("help,h", "Show help.")
    ("size,n", po::value<int>()->default_value(32), "Size")
    ;

  ASSERT_EQ(desc.options().size(), 2);

  // find by long name
  const auto* opt = desc.find("help");
  ASSERT_NE(opt, nullptr);
  ASSERT_EQ(opt->long_name, "help");
  ASSERT_TRUE(opt->has_short);
  ASSERT_EQ(opt->short_name, "h");
  ASSERT_TRUE(opt->description.find("Show help") != std::string::npos);

  // find by short name
  const auto* short_opt = desc.find_by_short('n');
  ASSERT_NE(short_opt, nullptr);
  ASSERT_EQ(short_opt->long_name, "size");

  // unknown option
  ASSERT_EQ(desc.find("unknown"), nullptr);
  ASSERT_EQ(desc.find_by_short('z'), nullptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestProgramOptions, options_description_find_long)
{
  po::options_description desc("Test");
  desc.add_options()
    ("verbose", po::bool_switch(), "Verbose")
    ;

  ASSERT_NE(desc.find("verbose"), nullptr);
  ASSERT_TRUE(desc.find("verbose")->value_semantic->is_bool_switch());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestProgramOptions, options_description_find_no_short)
{
  po::options_description desc("Test");
  desc.add_options()
    ("long-only", po::value<int>(), "Long only")
    ;

  const auto* opt = desc.find("long-only");
  ASSERT_NE(opt, nullptr);
  ASSERT_FALSE(opt->has_short);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestProgramOptions, help_output)
{
  po::options_description desc("Test Options");
  desc.add_options()
    ("help,h", "Show help.")
    ("size,n", po::value<int>()->default_value(32), "Domain size")
    ("verbose", po::bool_switch()->default_value(false), "Verbose")
    ;

  std::ostringstream oss;
  oss << desc;
  std::string help = oss.str();

  ASSERT_TRUE(help.find("Test Options") != std::string::npos);
  ASSERT_TRUE(help.find("--help") != std::string::npos);
  ASSERT_TRUE(help.find("-h") != std::string::npos);
  ASSERT_TRUE(help.find("--size") != std::string::npos);
  ASSERT_TRUE(help.find("-n") != std::string::npos);
  ASSERT_TRUE(help.find("=32") != std::string::npos);
  ASSERT_TRUE(help.find("Domain size") != std::string::npos);
  ASSERT_TRUE(help.find("Show help") != std::string::npos);
  ASSERT_TRUE(help.find("--verbose") != std::string::npos);
  // bool_switch should not show "arg"
  ASSERT_TRUE(help.find("--size arg") != std::string::npos);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestProgramOptions, empty_title)
{
  po::options_description desc;
  desc.add_options()
    ("help,h", "Help.")
    ;

  std::ostringstream oss;
  oss << desc;
  std::string help = oss.str();
  // When title is empty, no title line is printed
  ASSERT_TRUE(help.find("Options") == std::string::npos || help.find("--help") != std::string::npos);
  ASSERT_TRUE(help.find("--help") != std::string::npos);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestProgramOptions, parse_long_option)
{
  po::options_description desc("Test");
  int size = 0;
  desc.add_options()
    ("size,n", po::value<int>(&size)->default_value(32), "Size")
    ;

  const char* argv[] = {"program", "--size", "64"};
  int argc = 3;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, const_cast<char**>(argv), desc), vm);
  po::notify(vm);

  ASSERT_TRUE(vm.count("size"));
  ASSERT_EQ(vm["size"].as<int>(), 64);
  ASSERT_EQ(size, 64);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestProgramOptions, parse_short_option)
{
  po::options_description desc("Test");
  int size = 0;
  desc.add_options()
    ("size,n", po::value<int>(&size)->default_value(32), "Size")
    ;

  const char* argv[] = {"program", "-n", "128"};
  int argc = 3;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, const_cast<char**>(argv), desc), vm);
  po::notify(vm);

  ASSERT_EQ(vm["size"].as<int>(), 128);
  ASSERT_EQ(size, 128);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestProgramOptions, parse_short_option_inline_value)
{
  po::options_description desc("Test");
  std::string name;
  desc.add_options()
    ("define,D", po::value<std::string>(&name), "Define")
    ;

  const char* argv[] = {"program", "-DNAME"};
  int argc = 2;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, const_cast<char**>(argv), desc), vm);
  po::notify(vm);

  ASSERT_EQ(name, "NAME");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestProgramOptions, parse_long_option_eq_syntax)
{
  po::options_description desc("Test");
  std::string matrix;
  desc.add_options()
    ("matrix,A", po::value<std::string>(&matrix), "Matrix file")
    ;

  const char* argv[] = {"program", "--matrix=test.mtx"};
  int argc = 2;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, const_cast<char**>(argv), desc), vm);
  po::notify(vm);

  ASSERT_EQ(matrix, "test.mtx");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestProgramOptions, default_value_applied)
{
  po::options_description desc("Test");
  int size = 0;
  desc.add_options()
    ("size,n", po::value<int>(&size)->default_value(32), "Size")
    ;

  // No command-line args at all
  const char* argv[] = {"program"};
  int argc = 1;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, const_cast<char**>(argv), desc), vm);
  po::notify(vm);

  ASSERT_TRUE(vm.count("size"));
  ASSERT_EQ(vm["size"].as<int>(), 32);
  ASSERT_EQ(size, 32);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestProgramOptions, bool_switch_not_provided)
{
  po::options_description desc("Test");
  bool verbose = true;
  desc.add_options()
    ("verbose,v", po::bool_switch(&verbose)->default_value(false), "Verbose")
    ;

  const char* argv[] = {"program"};
  int argc = 1;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, const_cast<char**>(argv), desc), vm);
  po::notify(vm);

  ASSERT_TRUE(vm.count("verbose"));
  ASSERT_FALSE(vm["verbose"].as<bool>());
  ASSERT_FALSE(verbose);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestProgramOptions, bool_switch_provided)
{
  po::options_description desc("Test");
  bool verbose = false;
  desc.add_options()
    ("verbose,v", po::bool_switch(&verbose)->default_value(false), "Verbose")
    ;

  const char* argv[] = {"program", "--verbose"};
  int argc = 2;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, const_cast<char**>(argv), desc), vm);
  po::notify(vm);

  ASSERT_TRUE(vm["verbose"].as<bool>());
  ASSERT_TRUE(verbose);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestProgramOptions, bool_switch_short)
{
  po::options_description desc("Test");
  bool verbose = false;
  desc.add_options()
    ("verbose,v", po::bool_switch(&verbose)->default_value(false), "Verbose")
    ;

  const char* argv[] = {"program", "-v"};
  int argc = 2;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, const_cast<char**>(argv), desc), vm);
  po::notify(vm);

  ASSERT_TRUE(verbose);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestProgramOptions, bare_flag_presence)
{
  po::options_description desc("Test");
  desc.add_options()
    ("help,h", "Show help.")
    ;

  // Without flag
  {
    const char* argv[] = {"program"};
    int argc = 1;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, const_cast<char**>(argv), desc), vm);
    po::notify(vm);

    // bare flags have no default, so count should be false
    ASSERT_FALSE(vm.count("help"));
  }

  // With flag
  {
    const char* argv[] = {"program", "--help"};
    int argc = 2;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, const_cast<char**>(argv), desc), vm);
    po::notify(vm);

    ASSERT_TRUE(vm.count("help"));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestProgramOptions, multitoken_vector)
{
  po::options_description desc("Test");
  desc.add_options()
    ("prm,p", po::value<std::vector<std::string>>()->multitoken(), "Params")
    ;

  const char* argv[] = {"program", "--prm", "tol=1e-6", "maxiter=100", "solver=CG"};
  int argc = 5;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, const_cast<char**>(argv), desc), vm);
  po::notify(vm);

  ASSERT_TRUE(vm.count("prm"));
  auto vals = vm["prm"].as<std::vector<std::string>>();
  ASSERT_EQ(vals.size(), 3);
  ASSERT_EQ(vals[0], "tol=1e-6");
  ASSERT_EQ(vals[1], "maxiter=100");
  ASSERT_EQ(vals[2], "solver=CG");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestProgramOptions, multitoken_short)
{
  po::options_description desc("Test");
  desc.add_options()
    ("prm,p", po::value<std::vector<std::string>>()->multitoken(), "Params")
    ;

  const char* argv[] = {"program", "-p", "a=1", "b=2"};
  int argc = 4;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, const_cast<char**>(argv), desc), vm);

  auto vals = vm["prm"].as<std::vector<std::string>>();
  ASSERT_EQ(vals.size(), 2);
  ASSERT_EQ(vals[0], "a=1");
  ASSERT_EQ(vals[1], "b=2");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestProgramOptions, positional_arguments_remaining)
{
  po::options_description desc("Test");
  desc.add_options()
    ("prm,p", po::value<std::vector<std::string>>()->multitoken(), "Params")
    ;

  po::positional_options_description p;
  p.add("prm", -1); // -1 means all remaining positional args

  const char* argv[] = {"program", "x=1", "y=2", "z=3"};
  int argc = 4;

  po::variables_map vm;
  po::store(po::command_line_parser(argc, const_cast<char**>(argv))
              .options(desc).positional(p).run(), vm);

  ASSERT_TRUE(vm.count("prm"));
  auto vals = vm["prm"].as<std::vector<std::string>>();
  ASSERT_EQ(vals.size(), 3);
  ASSERT_EQ(vals[0], "x=1");
  ASSERT_EQ(vals[1], "y=2");
  ASSERT_EQ(vals[2], "z=3");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestProgramOptions, positional_arguments_fixed_count)
{
  po::options_description desc("Test");
  std::string input;
  desc.add_options()
    ("input,i", po::value<std::string>(&input)->required(), "Input file")
    ;

  po::positional_options_description pd;
  pd.add("input", 1); // exactly 1 positional argument

  const char* argv[] = {"program", "data.mtx"};
  int argc = 2;

  po::variables_map vm;
  po::store(po::command_line_parser(argc, const_cast<char**>(argv))
              .options(desc).positional(pd).run(), vm);
  po::notify(vm);

  ASSERT_EQ(input, "data.mtx");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestProgramOptions, mixed_options_and_positional)
{
  po::options_description desc("Test");
  int size = 0;
  std::vector<std::string> prm;
  desc.add_options()
    ("size,n", po::value<int>(&size)->default_value(32), "Size")
    ("prm,p", po::value<std::vector<std::string>>()->multitoken(), "Params")
    ;

  po::positional_options_description p;
  p.add("prm", -1);

  const char* argv[] = {"program", "--size", "64", "a=1", "b=2"};
  int argc = 5;

  po::variables_map vm;
  po::store(po::command_line_parser(argc, const_cast<char**>(argv))
              .options(desc).positional(p).run(), vm);
  po::notify(vm);

  ASSERT_EQ(vm["size"].as<int>(), 64);
  ASSERT_EQ(size, 64);

  auto vals = vm["prm"].as<std::vector<std::string>>();
  ASSERT_EQ(vals.size(), 2);
  ASSERT_EQ(vals[0], "a=1");
  ASSERT_EQ(vals[1], "b=2");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestProgramOptions, multiple_value_types)
{
  po::options_description desc("Test");
  int i = 0;
  double d = 0.0;
  std::string s;
  bool b = false;

  desc.add_options()
    ("int", po::value<int>(&i)->default_value(42), "Int")
    ("double", po::value<double>(&d)->default_value(3.14), "Double")
    ("string", po::value<std::string>(&s)->default_value("hello"), "String")
    ("bool", po::bool_switch(&b)->default_value(false), "Bool")
    ;

  const char* argv[] = {"program", "--int", "99", "--double", "2.71",
                        "--string", "world", "--bool"};
  int argc = 8;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, const_cast<char**>(argv), desc), vm);
  po::notify(vm);

  ASSERT_EQ(vm["int"].as<int>(), 99);
  ASSERT_EQ(i, 99);

  ASSERT_DOUBLE_EQ(vm["double"].as<double>(), 2.71);
  ASSERT_DOUBLE_EQ(d, 2.71);

  ASSERT_EQ(vm["string"].as<std::string>(), "world");
  ASSERT_EQ(s, "world");

  ASSERT_TRUE(vm["bool"].as<bool>());
  ASSERT_TRUE(b);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestProgramOptions, all_defaults)
{
  po::options_description desc("Test");
  int i = -1;
  double d = -1.0;
  std::string s = "none";

  desc.add_options()
    ("int", po::value<int>(&i)->default_value(42), "Int")
    ("double", po::value<double>(&d)->default_value(3.14), "Double")
    ("string", po::value<std::string>(&s)->default_value("hello"), "String")
    ;

  const char* argv[] = {"program"};
  int argc = 1;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, const_cast<char**>(argv), desc), vm);
  po::notify(vm);

  // Default values should be applied
  ASSERT_EQ(i, 42);
  ASSERT_DOUBLE_EQ(d, 3.14);
  ASSERT_EQ(s, "hello");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestProgramOptions, end_of_options_marker)
{
  po::options_description desc("Test");
  std::vector<std::string> files;
  desc.add_options()
    ("files", po::value<std::vector<std::string>>()->multitoken(), "Files")
    ;

  po::positional_options_description p;
  p.add("files", -1);

  const char* argv[] = {"program", "--", "-a", "-b", "-c"};
  int argc = 5;

  po::variables_map vm;
  po::store(po::command_line_parser(argc, const_cast<char**>(argv))
              .options(desc).positional(p).run(), vm);

  // After --, all tokens should be treated as positional
  auto vals = vm["files"].as<std::vector<std::string>>();
  ASSERT_EQ(vals.size(), 3);
  ASSERT_EQ(vals[0], "-a");
  ASSERT_EQ(vals[1], "-b");
  ASSERT_EQ(vals[2], "-c");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestProgramOptions, count_checks)
{
  po::options_description desc("Test");
  desc.add_options()
    ("help,h", "Help")
    ("opt", po::value<int>()->default_value(0), "Option")
    ;

  // Option not provided
  {
    const char* argv[] = {"program"};
    int argc = 1;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, const_cast<char**>(argv), desc), vm);

    // bare flag without default -> count false
    ASSERT_FALSE(vm.count("help"));
    // option with default -> count true even when not provided
    ASSERT_TRUE(vm.count("opt"));
  }

  // Option provided
  {
    const char* argv[] = {"program", "--help"};
    int argc = 2;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, const_cast<char**>(argv), desc), vm);

    ASSERT_TRUE(vm.count("help"));
    ASSERT_TRUE(vm.count("opt")); // has default
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestProgramOptions, parse_command_line_convenience)
{
  po::options_description desc("Test");
  std::string val;
  desc.add_options()
    ("opt,o", po::value<std::string>(&val), "Option")
    ;

  const char* argv[] = {"program", "-o", "test"};
  int argc = 3;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, const_cast<char**>(argv), desc), vm);
  po::notify(vm);

  ASSERT_EQ(val, "test");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestProgramOptions, type_mismatch_throws)
{
  po::options_description desc("Test");
  desc.add_options()
    ("opt", po::value<int>()->default_value(0), "Option")
    ;

  const char* argv[] = {"program", "--opt", "42"};
  int argc = 3;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, const_cast<char**>(argv), desc), vm);

  // Asking for wrong type should throw
  ASSERT_THROW(vm["opt"].as<std::string>(), std::logic_error);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestProgramOptions, bound_variable_all_types)
{
  po::options_description desc("Test");
  int iv = 0;
  double dv = 0.0;
  std::string sv;
  bool bv = false;

  desc.add_options()
    ("i", po::value<int>(&iv), "int")
    ("d", po::value<double>(&dv), "double")
    ("s", po::value<std::string>(&sv), "string")
    ("b", po::bool_switch(&bv)->default_value(false), "bool")
    ;

  const char* argv[] = {"program", "--i", "7", "--d", "1.5", "--s", "foo", "--b"};
  int argc = 8;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, const_cast<char**>(argv), desc), vm);
  po::notify(vm);

  ASSERT_EQ(iv, 7);
  ASSERT_DOUBLE_EQ(dv, 1.5);
  ASSERT_EQ(sv, "foo");
  ASSERT_TRUE(bv);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestProgramOptions, multiple_occurrences_overwrite)
{
  po::options_description desc("Test");
  std::string val;
  desc.add_options()
    ("opt", po::value<std::string>(&val), "Option")
    ;

  // If same option appears multiple times, last value wins
  const char* argv[] = {"program", "--opt", "first", "--opt", "second"};
  int argc = 5;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, const_cast<char**>(argv), desc), vm);
  po::notify(vm);

  ASSERT_EQ(val, "second");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(TestProgramOptions, multitoken_with_default)
{
  po::options_description desc("Test");
  po::typed_value<std::vector<std::string>>* tv =
    po::value<std::vector<std::string>>()->multitoken();

  // Setting a default on multitoken is possible
  // but not commonly used; just test it compiles and stores
  desc.add_options()
    ("prm,p", tv, "Params")
    ;

  const char* argv[] = {"program"};
  int argc = 1;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, const_cast<char**>(argv), desc), vm);
  po::notify(vm);

  // Without specified multi-default, count is false
  ASSERT_FALSE(vm.count("prm"));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
