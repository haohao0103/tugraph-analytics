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

CREATE TABLE tbl_result (
  a_id bigint,
  b_id bigint,
  c_id bigint,
  d_id bigint,
  f_id bigint,
  g_id bigint
) WITH (
	type='file',
	geaflow.dsl.file.path='${target}'
);

USE GRAPH modern;

INSERT INTO tbl_result
SELECT
	a.id,
	b.id,
	c.id,
	d.id,
	f.id,
	g.id
FROM (
  MATCH
       (a:person where a.id = 4)-[e:created]->(b)
       | (c:person where c.id = 4)-[]->(d)<-()
       | (f:person where f.id = 4) -[]- (g)
  RETURN a, b, c, d, f, g
)
;

