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

package com.antgroup.geaflow.example.stream;

import com.antgroup.geaflow.api.function.internal.CollectionSource;
import com.antgroup.geaflow.api.function.io.SinkFunction;
import com.antgroup.geaflow.api.pdata.stream.window.PWindowSource;
import com.antgroup.geaflow.api.window.WindowFactory;
import com.antgroup.geaflow.api.window.impl.SizeTumblingWindow;
import com.antgroup.geaflow.common.config.Configuration;
import com.antgroup.geaflow.env.Environment;
import com.antgroup.geaflow.env.EnvironmentFactory;
import com.antgroup.geaflow.example.util.ExampleSinkFunctionFactory;
import com.antgroup.geaflow.pipeline.Pipeline;
import com.antgroup.geaflow.pipeline.PipelineFactory;
import com.antgroup.geaflow.pipeline.task.IPipelineTaskContext;
import com.antgroup.geaflow.pipeline.task.PipelineTask;
import com.google.common.collect.Lists;
import java.util.List;

public class WindowStreamWordCount {

    public static void main(String[] args) {
        Environment environment = EnvironmentFactory.onRayEnvironment(args);
        Pipeline pipeline = PipelineFactory.buildPipeline(environment);
        pipeline.submit(new PipelineTask() {
            @Override
            public void execute(IPipelineTaskContext pipelineTaskCxt) {
                Configuration config = pipelineTaskCxt.getConfig();
                List<String> words = Lists.newArrayList("hello", "world", "hello", "word");
                PWindowSource<String> streamSource =
                    pipelineTaskCxt.buildSource(new CollectionSource<String>(words) {},
                        SizeTumblingWindow.of(100));

                SinkFunction<String> sink = ExampleSinkFunctionFactory.getSinkFunction(config);
                streamSource.window(WindowFactory.createSizeTumblingWindow(2))
                    .sink(sink);
            }
        });

        pipeline.execute();
    }
}
