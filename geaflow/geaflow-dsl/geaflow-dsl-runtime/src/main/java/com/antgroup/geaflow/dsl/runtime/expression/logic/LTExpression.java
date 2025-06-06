/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.antgroup.geaflow.dsl.runtime.expression.logic;

import com.antgroup.geaflow.common.type.Types;
import com.antgroup.geaflow.dsl.runtime.expression.AbstractReflectCallExpression;
import com.antgroup.geaflow.dsl.runtime.expression.Expression;
import com.antgroup.geaflow.dsl.schema.function.GeaFlowBuiltinFunctions;
import com.google.common.collect.Lists;
import java.util.List;

public class LTExpression extends AbstractReflectCallExpression {

    public static final String METHOD_NAME = "lessThan";

    public LTExpression(Expression left, Expression right) {
        super(Lists.newArrayList(left, right), Types.BOOLEAN, GeaFlowBuiltinFunctions.class, METHOD_NAME);
    }

    @Override
    public String showExpression() {
        return inputs.get(0).showExpression() + " < " + inputs.get(1).showExpression();
    }

    @Override
    public Expression copy(List<Expression> inputs) {
        assert inputs.size() == 2;
        return new LTExpression(inputs.get(0), inputs.get(1));
    }
}
