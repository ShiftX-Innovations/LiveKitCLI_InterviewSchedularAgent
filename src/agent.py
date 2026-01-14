import logging
from urllib.parse import quote
from typing import Optional
import aiohttp
import asyncio
import json
from pybars import Compiler
from dotenv import load_dotenv
from livekit import rtc, api, agents
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    ToolError,
    cli,
    function_tool,
    inference,
    room_io,
    utils,
    get_job_context,
    llm
)
from livekit.plugins import (
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from datetime import datetime, UTC
from dataclasses import dataclass, field
import httpx
from utils.utils import load_prompt

logger = logging.getLogger("google-interview-agent")

load_dotenv(".env.local")


class VariableTemplater:
    def __init__(self, metadata: str, additional: dict[str, dict[str, str]] | None = None) -> None:
        self.variables = {
            "metadata": self._parse_metadata(metadata),
        }
        if additional:
            self.variables.update(additional)
        self._cache = {}
        self._compiler = Compiler()

    def _parse_metadata(self, metadata: str) -> dict:
        try:
            value = json.loads(metadata)
            if isinstance(value, dict):
                return value
            else:
                logger.warning(f"Job metadata is not a JSON dict: {metadata}")
                return {}
        except json.JSONDecodeError:
            return {}

    def _compile(self, template: str):
        if template in self._cache:
            return self._cache[template]
        self._cache[template] = self._compiler.compile(template)
        return self._cache[template]

    def render(self, template: str):
        return self._compile(template)(self.variables)


class DefaultAgent(Agent):
    def __init__(self, metadata: str) -> None:
        templater = VariableTemplater(metadata)

        raw_prompt = load_prompt("initial_agent_prompt.yaml")
        rendered_prompt = templater.render(raw_prompt)

        self._templater = templater
        self._start_time = datetime.now()
        self.collected_data = {
            "transcript": "",
            "customerAnswers": "{}", 
            "qualify_status": False,
            "interview_scheduled_date": "",
            "interview_scheduled_time": "",
            "call_status": "in-progress"
        }
        
        super().__init__(
            instructions=rendered_prompt,
            )
        
        
    async def trigger_end_call_api(self, room_name: str,reason: str):
        """Standardizes the API call for both the Tool and Lifecycle events."""
        if self._api_called:
            return
        self._api_called = True
        
        headers = {
            "Content-Type": "application/json",
        }
        payload = {
            "roomName": room_name,
            "agentID": "interview-agent", # Or get from ctx
            "transcript": "",
            "endedReason": reason,
            "successEvaluation": "success",
            "customerAnswers": "",
            "qualified": self.collected_data["qualify_status"],
            "call_status": self.collected_data.get("call_status", "not_scheduled"),
            "interview_scheduled_date": self.collected_data.get("interview_scheduled_date", ""),
            "interview_scheduled_time": self.collected_data.get("interview_scheduled_time", ""),
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post("https://hr.meetvoxa.ai/api/calls/webhook/end-meeting",headers=headers, json=payload) as resp:
                    if resp.status >= 400:
                        logger.warning(f"Webhook failed: {resp.status}")
                        
            current_speech = self.session.current_speech
            if current_speech:
                await current_speech.wait_for_playout()

            # 4. Trigger the actual room deletion
            # await self.hangup()
            
            return "Meeting ended successfully."
                
        except Exception as e:
            logger.error(f"Error in end_call tool: {e}")
            raise ToolError(f"Failed to end call: {e}")
        
    async def on_enter(self):
        await self.session.generate_reply(
            instructions=self._templater.render("""Greet the user and offer your assistance."""),
            allow_interruptions=True,
        )

    @function_tool(name="update_candidate_info")
    async def update_candidate_info(
        self, 
        context: agents.RunContext, 
        qualify_status: bool = None,
        call_status: str = None,
        interview_scheduled_date: str = None,
        interview_scheduled_time: str = None
    ):
        """
        Call this tool as soon as the candidate qualified for interview or successfully scheduled a interview or candidate accept a interview date and time.
        """
        if qualify_status:
            self.collected_data["qualify_status"] = qualify_status
        if call_status:
            self.collected_data["call_status"] = call_status
        if interview_scheduled_date:
            self.collected_data["interview_scheduled_date"] = interview_scheduled_date
        if interview_scheduled_time:
            self.collected_data["interview_scheduled_time"] = interview_scheduled_time
            
        logger.info(f"Updated candidate state: {self.collected_data}")
        return "Information saved successfully."

    @function_tool(name="end_call")
    async def _http_tool_end_call(
        self,
        context: RunContext, roomName: str, agentID: str, transcript: str, recordingUrl: str, startedAt: str, endedAt: str, endedReason: str, durationMs: float, successEvaluation: str, customerAnswers: str, qualified: bool, call_status: str, interview_scheduled_date:str, interview_scheduled_time:str,
        ctx: RunContext
    ) -> str:
        """
        End the call and send all data. ALWAYS call at the end when the user says thank you or goodbye.

        Args:
            roomName: 
            agentID: 
            transcript: 
            recordingUrl: 
            startedAt: 
            endedAt: 
            endedReason: 
            durationMs: 
            successEvaluation:  
            customerAnswers: 
            qualified: 
            call_status: 
            interview_scheduled_date:
            interview_scheduled_time:
        """
        self._api_called = True
        context.disallow_interruptions()

        url = "https://hr.meetvoxa.ai/api/calls/webhook/end-meeting"
        headers = {
            "Content-Type": "application/json",
        }
        payload = {
            "roomName": roomName,
            "agentID": agentID,
            "transcript": transcript,
            "recordingUrl": recordingUrl,
            "startedAt": startedAt,
            "endedAt": endedAt,
            "endedReason": endedReason,
            "durationMs": durationMs,
            "successEvaluation": successEvaluation,
            "customerAnswers": customerAnswers,
            "qualified": qualified,
            "call_status": call_status,
            "interview_scheduled_date": interview_scheduled_date,
            "interview_scheduled_time": interview_scheduled_time,
        }

        try:
            session = utils.http_context.http_session()
            async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status >= 400:
                        logger.warning(f"Webhook failed with status {resp.status}")

            current_speech = self.session.current_speech
            if current_speech:
                await current_speech.wait_for_playout()

            # 4. Trigger the actual room deletion
            # await self.hangup()
            
            return "Meeting ended successfully."
                
        except Exception as e:
            logger.error(f"Error in end_call tool: {e}")
            raise ToolError(f"Failed to end call: {e}")
    
    @function_tool(name="get_meeting_details")
    async def _http_tool_get_meeting_details(
        self, context: RunContext, externalId: str
    ) -> str:
        """
        Initially Get meeting details at the start. Call this FIRST before speaking

        Args:
            externalId: {{metadata.external_id}}
        """

        # url = f"https://hr.meetvoxa.ai/api/calls/get-meeting-details/{quote(externalId, safe='')}"
        url = f"https://hr.meetvoxa.ai/api/calls/get-meeting-details/8cf9b39a-4534-413e-91f8-b13fbc02e1ce"

        try:
            session = utils.http_context.http_session()
            timeout = aiohttp.ClientTimeout(total=10)
            async with session.get(url, timeout=timeout) as resp:
                body = await resp.text()
                if resp.status >= 400:
                    raise ToolError(f"error: HTTP {resp.status}: {body}")
                logger.info(
                    "[LiveKit Tool] HTTP success | url=%s",
                    body,
                )
                import json
                try:
                    data = json.loads(body)
                    # Store it in your agent instance
                    self.collected_data["meeting_details"] = data 
                    logger.info("Meeting details saved to agent state.")
                except Exception as e:
                    logger.error(f"Failed to parse meeting details: {e}")
                return body
        except ToolError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise ToolError(f"error: {e!s}") from e

    @function_tool()
    async def transfer_to_schedular(self, context: RunContext):
        """Transfer the customer to a billing specialist for account and payment questions."""
        return MeetingSchedularAgent(chat_ctx=self.chat_ctx), "Transferring to schedular"

class MeetingSchedularAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are a meeting schedular. Guide qualified candidate to schedule a meeting by suggesting available time slots. Be thorough and empathetic."""
        )
    async def on_enter(self) -> None:
        await self.session.generate_reply(instructions="Tell Candidate to that he has qualified for the second interview and suggest candidate to availble time slot")

    @function_tool(name="schedule_interview")
    async def _http_tool_schedule_interview(
        self, context: RunContext, external_id: str, start_time: str
    ) -> str:
        """
        Schedule interview for qualified candidates.

        Args:
            external_id: 
            start_iso: candidate accepted date time in iso format
        """

        context.disallow_interruptions()

        url = "https://hr.meetvoxa.ai/api/calendar-integrations/public/interviews/schedule"
        payload = {
            "match_id": external_id,
            "start_iso": start_time
        }

        try:
            session = utils.http_context.http_session()
            timeout = aiohttp.ClientTimeout(total=10)
            async with session.post(url, timeout=timeout, json=payload) as resp:
                body = await resp.text()
                if resp.status >= 400:
                    raise ToolError(f"error: HTTP {resp.status}: {body}")
                
                self.collected_data["call_status"] = "scheduled"
                
                return body
        except ToolError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise ToolError(f"error: {e!s}") from e

    @function_tool(name="get_slot")
    async def _http_tool_get_slot(
            self, context: RunContext, job_id: str,
        ) -> str:
            """
            Get latest interview slots for qualified candidates and suggest them earliest avaiable slot.

            Args:
                job_id: [job_id]
            """

            context.disallow_interruptions()
            headers = {
                        "Content-Type": "application/json",
                    }
            # url = "https://voxahr-api.shiftx.tech/api/calendar-integrations/jobs/31c8298b-7170-4ca6-a655-ab54b7127a5a/available-slots"
            url = f"https://voxahr-api.shiftx.tech/api/calendar-integrations/jobs/{quote(job_id, safe='')}/available-slots"
            payload = {
                "provider": "google"
            }

            try:
                session = utils.http_context.http_session()
                timeout = aiohttp.ClientTimeout(total=10)
                async with session.post(url, timeout=timeout,headers=headers, json=payload) as resp:
                    body = await resp.text()
                    if resp.status >= 400:
                        raise ToolError(f"error: HTTP {resp.status}: {body}")
                    logger.info(f"available slots: {body}")
                    return body
                    
            except ToolError:
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                raise ToolError(f"error: {e!s}") from e


server = AgentServer()

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

server.setup_fnc = prewarm

async def on_session_end(ctx: JobContext) -> None:
    report = ctx.make_session_report()
    report_dict = report.to_dict()
    chat_history = report_dict["chat_history"]
    
    items = chat_history.get("items", [])
    filtered_items = [item for item in items if item.get("type") == "message"]
    filtered_transcript = {"items": filtered_items}
    
    payload = {
        "roomName": ctx.room.name,
        "transcript" : filtered_transcript,
        "sessionReport": report_dict,
        "receivedAt": datetime.now(UTC).isoformat(),
        "agentData": ctx.proc.userdata.get("agent_data"),
    }

    async with httpx.AsyncClient(timeout=10) as client:
        await client.post(
            "https://hr.meetvoxa.ai/api/calls/transcript",
            json=payload,
        )

    print(f"Session report for {ctx.room.name} saved to {payload}")

@server.rtc_session(agent_name="google-agent",on_session_end=on_session_end)
async def entrypoint(ctx: JobContext):
    room_name = ctx.room.name
    
    agent = DefaultAgent(metadata=ctx.job.metadata)
    ctx.proc.userdata["agent_data"] = agent.collected_data
    
    async def final_cleanup():
        await agent.trigger_end_call_api(room_name=room_name, reason="Call_Ended_By_Candidate")

    ctx.add_shutdown_callback(final_cleanup)

    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="en"),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=inference.TTS(
            model="elevenlabs/eleven_flash_v2_5",
            voice="Xb7hH8MSUJpSbSDYk0k2",
            language="en"
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True
    )

    await session.start(
        agent=agent,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony() if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP else noise_cancellation.BVC(),
            ),
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
