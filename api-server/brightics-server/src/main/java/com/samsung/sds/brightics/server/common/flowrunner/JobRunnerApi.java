package com.samsung.sds.brightics.server.common.flowrunner;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.samsung.sds.brightics.common.workflow.context.parameter.Parameters;
import com.samsung.sds.brightics.common.workflow.runner.AbsJobRunnerApi;
import com.samsung.sds.brightics.common.workflow.runner.vo.JobParam;
import com.samsung.sds.brightics.common.workflow.runner.vo.JobStatusVO;
import com.samsung.sds.brightics.server.common.holder.BeanHolder;
import com.samsung.sds.brightics.server.common.message.task.TaskMessageBuilder;
import com.samsung.sds.brightics.server.common.message.task.TaskMessageRepository;

public class JobRunnerApi extends AbsJobRunnerApi {

	@Override
	public String executeTask(String taskId, String name, String parameters, String attributes) {
		BeanHolder.getBeanHolder().taskService.executeSyncTask(TaskMessageBuilder.newBuilder(taskId, name)
				.setAttributes(attributes).setParameters(parameters).build());
		return taskId;
	}

	@Override
	public boolean isFinishTask(String taskId) {
		return TaskMessageRepository.isExistFinishMessage(taskId);
	}

	@Override
	public Object getTaskResult(String taskId) {
		return BeanHolder.getBeanHolder().messageManager.taskManager().getAsyncTaskResult(taskId);
	}

	@Override
	public void stopTask(String stopTaskId, String functionName, String context) {
		BeanHolder.getBeanHolder().taskService.stopTask(stopTaskId, functionName, context);
	}

	@Override
	public Object getDatasourceInfo(String name) {
		return BeanHolder.getBeanHolder().dataSourceService.getDatasourceInfo(name);
	}

	@Override
	public String getScriptWithParam(Parameters params) {
		return BeanHolder.getBeanHolder().pyFunctionService.getScriptWithParam(params);
	}

	@Override
	public boolean isMetadataRequest(JsonObject json) {
		return BeanHolder.getBeanHolder().metadataConverterService.isMetadataRequest(json);
	}

	@Override
	public JsonElement convert(JsonObject json) {
		return BeanHolder.getBeanHolder().metadataConverterService.convert(json);
	}

	@Override
	public void updateJobStatus(JobParam jobParam, JobStatusVO jobStatusVO) {
		BeanHolder.getBeanHolder().jobStatusService.updateJobStatus(jobParam, jobStatusVO);

	}

	@Override
	public Object getData(String mid, String tid, long min, long max) {
		return BeanHolder.getBeanHolder().dataService.getData(mid, tid, min, max);
	}

	@Override
	public void addDataAlias(String source, String alias) {
		BeanHolder.getBeanHolder().dataService.addDataAlias(source, alias);
	}

}