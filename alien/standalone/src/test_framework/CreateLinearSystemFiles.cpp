/*
* Copyright 2020 IFPEN-CEA
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* SPDX-License-Identifier: Apache-2.0
*/

#include <string>
#include <fstream>

#include "CreateLinearSystemFiles.h"

void createMMMatrixFile(std::string const& file_name)
{
  const std::string mat =
  "%%MatrixMarket matrix coordinate real general\n"
  "%-------------------------------------------------------------------------------\n"
  "% UF Sparse Matrix Collection, Tim Davis\n"
  "% http://www.cise.ufl.edu/research/sparse/matrices/vanHeukelum/cage4\n"
  "% name: vanHeukelum/cage4\n"
  "% [DNA electrophoresis, 4 monomers in polymer. A. van Heukelum, Utrecht U.]\n"
  "% id: 905\n"
  "% date: 2003\n"
  "% author: A. van Heukelum\n"
  "% ed: T. Davis\n"
  "% fields: title A name id date author ed kind\n"
  "% kind: directed weighted graph\n"
  "%-------------------------------------------------------------------------------\n"
  "9 9 49\n"
  "1 1 .75\n"
  "2 1 .075027667114587\n"
  "4 1 .0916389995520797\n"
  "5 1 .0375138335572935\n"
  "8 1 .0458194997760398\n"
  "1 2 .137458499328119\n"
  "2 2 .687569167786467\n"
  "3 2 .0916389995520797\n"
  "5 2 .0375138335572935\n"
  "6 2 .0458194997760398\n"
  "2 3 .112541500671881\n"
  "3 3 .666666666666667\n"
  "4 3 .13745849932812\n"
  "6 3 .0458194997760398\n"
  "7 3 .0375138335572935\n"
  "1 4 .112541500671881\n"
  "3 4 .075027667114587\n"
  "4 4 .729097498880199\n"
  "7 4 .0375138335572935\n"
  "8 4 .0458194997760398\n"
  "1 5 .137458499328119\n"
  "2 5 .075027667114587\n"
  "5 5 .537513833557293\n"
  "6 5 .075027667114587\n"
  "7 5 .0916389995520797\n"
  "9 5 .0833333333333333\n"
  "2 6 .112541500671881\n"
  "3 6 .0916389995520797\n"
  "5 6 .13745849932812\n"
  "6 6 .445874834005214\n"
  "8 6 .13745849932812\n"
  "9 6 .075027667114587\n"
  "3 7 .075027667114587\n"
  "4 7 .13745849932812\n"
  "5 7 .112541500671881\n"
  "7 7 .470791832661453\n"
  "8 7 .112541500671881\n"
  "9 7 .0916389995520797\n"
  "1 8 .112541500671881\n"
  "4 8 .0916389995520797\n"
  "6 8 .075027667114587\n"
  "7 8 .0916389995520797\n"
  "8 8 .54581949977604\n"
  "9 8 .0833333333333333\n"
  "5 9 .25\n"
  "6 9 .150055334229174\n"
  "7 9 .183277999104159\n"
  "8 9 .25\n"
  "9 9 .166666666666667\n";

  {
    std::fstream matrix_file_stream(file_name, std::ios_base::out);
    matrix_file_stream << mat;
  }
}

void createMMRhsFile(std::string const& file_name)
{
  const std::string rhs =
  "%%MatrixMarket matrix array real general\n"
  "%-------------------------------------------------------------------------------\n"
  "% Fake rhs for test\n"
  "%-------------------------------------------------------------------------------\n"
  "9 1\n"
  ".75\n"
  ".075027667114587\n"
  ".0916389995520797\n"
  ".0375138335572935\n"
  ".0458194997760398\n"
  ".137458499328119\n"
  ".687569167786467\n"
  ".0916389995520797\n"
  ".0375138335572935\n";

  {
    std::fstream rhs_file_stream(file_name, std::ios_base::out);
    rhs_file_stream << rhs;
  }
}

void createSSArchive(std::string const& base_name)
{
  const std::string mat =
  "%%MatrixMarket matrix coordinate real general\n"
  "%-------------------------------------------------------------------------------\n"
  "% UF Sparse Matrix Collection, Tim Davis\n"
  "% http://www.cise.ufl.edu/research/sparse/matrices/Grund/b1_ss\n"
  "% name: Grund/b1_ss\n"
  "% [Unsymmetric Matrix b1_ss, F. Grund, Dec 1994.]\n"
  "% id: 449\n"
  "% date: 1997\n"
  "% author: F. Grund\n"
  "% ed: F. Grund\n"
  "% fields: title A b name id date author ed kind\n"
  "% kind: chemical process simulation problem\n"
  "%-------------------------------------------------------------------------------\n"
  "7 7 15\n"
  "5 1 -.03599942\n"
  "6 1 -.0176371\n"
  "7 1 -.007721779\n"
  "1 2 1\n"
  "2 2 -1\n"
  "1 3 1\n"
  "3 3 -1\n"
  "1 4 1\n"
  "4 4 -1\n"
  "2 5 .45\n"
  "5 5 1\n"
  "3 6 .1\n"
  "6 6 1\n"
  "4 7 .45\n"
  "7 7 1\n";

  const std::string rhs =
  "%%MatrixMarket matrix array real general\n"
  "%-------------------------------------------------------------------------------\n"
  "% UF Sparse Matrix Collection, Tim Davis\n"
  "% http://www.cise.ufl.edu/research/sparse/matrices/Grund/b1_ss\n"
  "% name: Grund/b1_ss : b matrix\n"
  "%-------------------------------------------------------------------------------\n"
  "7 1\n"
  "-.0001\n"
  ".1167\n"
  "-.2333\n"
  ".1167\n"
  "-.4993128\n"
  ".3435885\n"
  ".7467878\n";

  system(("rm -rf " + base_name).c_str());
  system(("mkdir " + base_name).c_str());
  {
    std::fstream mat_file_stream(base_name + "/" + base_name + ".mtx", std::ios_base::out);
    mat_file_stream << mat;
    std::fstream rhs_file_stream(base_name + "/" + base_name + "_b.mtx", std::ios_base::out);
    rhs_file_stream << rhs;
  }

  system(("tar -zcf " + base_name + ".tar.gz " + base_name).c_str());
  system(("rm -r " + base_name).c_str());
}