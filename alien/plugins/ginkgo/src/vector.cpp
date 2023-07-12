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

#include "vector.h"

#include <alien/ginkgo/backend.h>
#include <alien/ginkgo/machine_backend.h>
#include <alien/core/impl/MultiVectorImpl.h>

namespace Alien::Ginkgo
{
Vector::Vector(const MultiVectorImpl* multi_impl)
: IVectorImpl(multi_impl, AlgebraTraits<BackEnd::tag::ginkgo>::name())
, gko::matrix::Dense<double>(
  Ginkgo_executor::exec_map.at(Ginkgo_executor::target_machine)(), // throws if not valid
  gko::dim<2>(multi_impl->space().size(), 1))
, data(gko::dim<2>(multi_impl->space().size(), 1))
{
  alien_debug([&] {
    cout() << "Vector size : "
           << multi_impl->space().size();
    cout() << "data size : "
           << data.get_size()[0] << "row "
           << " - " << data.get_size()[1] << " cols";
  });
}

Vector::~Vector()
{
}

void Vector::setProfile(int ilower, int iupper)
{
}

void Vector::setValues(Arccore::ConstArrayView<double> values)
{
  auto ncols = values.size();
  for (auto icol = 0; icol < ncols; ++icol) {
    data.add_value(icol, 0, values[icol]);
  }
}

void Vector::getValues(Arccore::ArrayView<double> values) const
{
  // get internal ginkgo values, and write them into values array.
  const double* ginkgo_values = this->get_const_values();

  int icol = data.get_size()[0];

  for (int i = 0; i < icol; i++) {
    values.setAt(i, ginkgo_values[i]);
  }
}

void Vector::assemble()
{
  if ((this->get_size()[0] == data.get_size()[0]) && (this->get_size()[1] == data.get_size()[1])) {
    this->read(data);
  }
  else
    throw Arccore::FatalErrorException("Vec size does not match data size");
}

} // namespace Alien::Ginkgo
