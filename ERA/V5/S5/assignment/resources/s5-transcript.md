# Session 5 (ERA V5) — Live Class Transcript

Source: Google Meet, transcribed via [Tactiq](https://tactiq.io). Captured from
https://app.tactiq.io/transcripts/Nd1BQPd9BFEyZaNva0lw/transcript

Recorded Sat, 25/07/2026 · 147 min · 31 participants (The Admin/Rohan ~83% of talk time).

---

TA
The Admin
00:00
Open this video, which is dot m-o-g. Emoji. Let me show that, okay, that may also exist. Why not do a Maggie let's? Let's do a file called four dot Maggie, and uh, already change the FPS or remove the last 25 frames. Let's say this kind of command is there, right? And we run this command actually in the environment, and the law comes back and say that dot m a g g is not supported.
00:26
We will not be training on this, but will sending intermodels the model can look at it and say, ah, it's not there. So, you can ask the user your format is wrong. Right, or it will. You can remember that Maggie is not a format to be. To run on ffmpx, you can. You can make those destinations. So, where we are going with this is, as we continue to train, you're going to realize that. Okay, here we have some interesting people.
00:50
Susan Boyle, Charles Darwin, Stephen Colbert, Celine Dion, and others, right? So? So, and that's where the data mixture comes in. Everything that we collect is not. Are put in the loss again. What is loss? Loss is. Whether we are telling whether we are punishing the model for not predicting a particular token, right? So, now, that should make sense. We should not punish
💬
Vardhan Walavalkar
01:10
Hi everyone, this is an automated message to let you know my Tactiq extension: https://tactiq.io/r/transcribing is transcribing this meeting for me so I can give my full attention to you.
TA
The Admin
01:13
the model for predicting what any random tool can give us as an output, right? So, when I run on my application, for example, this camera application? Something I only build. Now when I run this, I get this kind of log. Right, there's no point training on all of this. We will send this to the model we'll send to the model to understand. Okay, if I launch this, what happens? What all can we launch? What is the kind of data that comes back, but will not be training on it. It's very important and like this, uh, this will get crystallized as we discuss more today. The point is that we need to have a training plan based on a training plan? We are going to decide on the data mixture, what kind of data we are going to be mixing at, what time.
💡
Callout
01:59
At each time. What category is the model going to see? For example, we discuss this agentic Trace. This is exactly the data you guys are creating right now. Which is your widget creation and talking niclifier? You did something your port was locked, right? What's the point of training the model on a data where it says your port was locked, but it is important to see because model can see the portfolks log. This is what? Do what I should have done so model. Think okay, the port post log. I'll try and use another port instead of that. If it comes back to you and say that Otis log, I can't do anything. Then, we submit a log.
💡
Callout
✅
Action Item
02:33
Except with the loss right now, how often it sees a particular kind of data set at what stage the training, uh, it's going to be seen the clean coppers and the same compute budget can provide completely different kind of models, depending on when and when exactly. Are you feeding it? Right, because if you feed the high quality data which is on PhD, level, agentic traces, and? Function calling and.
02:56
Uh, latex and clear python scope or clear python books very initial to the model. When your model doesn't know where the model is doing, saying about a bunch of path, and it is just when you consume it without learning anything. So, we have to have a very clear idea of what part of the data we are going to be feeding when and if that is? Properly after the data is clean, then you, you guys, are at a very, very good model. Now, our Target is clear, and this is something that you and I need to align on.
03:24
Because because you guys are going to be collecting, uh, creating clearing data sets and? Allow us to make this model. Our model needs to be really good at really, really good at coding at the agentic task, right? When we say agent task, you saw what that Celine Dion is is intake. When you ask a question today, models are very clear that I don't know the answer. I'll go on online. I'll search. Where do I search? Important question, right? How do I search? What part of the search layer should I show all those questions is? What makes even a good chat chat part. Twitter chatbot is not ready after questioning response back. Every time you're going to be asking something, it goes online for a good deep research and comes back, and distance information gives it to you. So, without coding an agentic, you can't have any model that is useful. You can't come back and say that. Okay, I can make a financial model. What does it mean financial model would mean that you again have to have a tool a calling capability that you can call accelerated tools or a Max and Min and mode and averages and average with or running off
💡
Callout
04:21
with banking, banking order or? Right, running off in the banking format. So there are a lot of things that the model need to know, and for that, it needs to be able to use the functions or tools or calculations. You should be able to do a very long task. That is something that is an additional objective for us, which is what you see. Um, nowadays, it's getting less. But if you were using coding tools in 24 and 25, 25, then you will see that a model after some time, it starts behaving that I'm today's Friday evening. I'm going. It will just not respond, right? Sometimes the chart will just end sometimes RGBT, which is just just not load and continue. Because the long Horizon tasks were very difficult, and this is where.
05:04
The gold data is this is literally where the anti-matter data is there and laps are preserving it. If somehow we can collect this data? Then you. Again, a model can beat things and same thing here. If we look at? The pins that I opened earlier opened Opus 5. Asian determinal bench fine fine fine? Frontier bench, right? Now, cost per attempt. This is basically long Horizon part. First of, all these tasks are difficult, but how long does it take to actually solve a particular task. You see that performance goes from 25 to 45 percent to the cost cost of five dollars to twenty dollar.
05:46
Right? So this is, this is something that needs to be fixed. This is something that. Uh, you have Arcadia again. AJI kind of problems where Opus is very low, but here we have Opus 5 again very high, but the cost will see is 20 000 for solving this Benchmark, right? If you were to take part in this Benchmark, that amount of money? We'll have to spend way more because this is the deployed Opus course, which is way less if you were to do this on the.
06:13
Test level or? The model is getting cleaned on the big H200. Probably will have to spend $100,000 just on this particular evaluation. All right, so all these evaluation essentially humanities last exam. For example, here again, you're seeing more cost being spent. The point is that you can actually make an end-to-end application. This is an example of what Opus 5 built. Now, this is a working example so we can open it and check it out.
06:39
Right. So here you can see that this whole application is built by one chat bot one single prompt, so it needs to keep on thinking you need to keep on learning things. Keep on checking things you need to know how to check things and what kind of. Data might be required for the user, interact with how does the UI look like all of that? Information needs to go in.
07:01
That is why this long horror isn't is one of the I'll say Mecca of things because you can put an AI which or agent or model in our case to be able to continue to think and continue to improve and continue to solve its own mistakes. You essentially are solving what India India doesn't have as a country, right? Okay, now the target is clear for us, the coding, and the agentic work is the utmost importance. It should be able to take a long task plan, the work.
07:29
Uh, call tools across multiple steps. Read the results, then recover the call fails and continue, and this is exactly where I was talking about. Now, here you'll see that we are saying recur when a call fails recover when a call fails. The model did something. For example, launch a particular function or call a function in a wrong way or write a wrong code. The compiler fails or the, the on python error comes in and based on the error, nor the model does something new.
07:56
So, while training that error, we do not train on. We will send it to the model, so model can see what error was, but we are not going to train the model to become python. Training on the error log would mean that the model can think or can work like a python compiler, which is a nonsense. Which is we, we don't need that. You can Define you can decide to make a model for that purpose only. But how error looks like, or how to looks like and writing that error exactly in the same way is not required. For example, when you.
💡
Callout
08:27
Do something something crashes, you know? It crashed. You can look at the important information and then proceed. For example, if this browser crashed or this website crashed, you'll get an error log. You'll know that, okay, this particular. Command and work, or it couldn't read that particular point from data set. That's the only important part rest. All the logs that are there that this was a time? This was a JavaScript version you're using. The series of the other applications or other extension that all the information are nonsense. So that's why we don't train on it. It's going to be a waste of time.
💡
Callout
08:53
But this is the objective. So if you ever know you're thinking of what kind of model we are planning, the target is clear. Coding and agentic, uh, need to be good at intake, of course. We have a codec style assistant, which is able to take a long task, plan the work, call the tools across multiple steps, read the results, recover when the call face, and continue operating while holding the growth, growing history of the tasking context. So that's the objective of the random model we are targeting. We also want to strong reasoning.
09:19
Whose death to reason can be controlled. They're very, very important. Now, everywhere, you would have seen that how deep you want the model to go. You wanted to think less. You wanted to think more. You wanted to think very high number of tokens, and that is something that you have to train on.
💡
Callout
09:32
Have you thought about it? How? How do you get the medium hard and super and Ultra thinking performance? The model has to be trained on. You can't just say I'll just use the first finder token and call it medium. I will just use the 5000 tokens and I'll call it. I'll just add, think more, think more, think more. That is not how it use. That's how it started, but that's not the state of the art. Now, you have to hint the model that I want to think more example. This is my favorite interview question. Whenever I hire people, I, they're explaining. I asked them to talk about yourself. They're talking about.
10:02
Got the date blah blah blah, then immediately I stopped them and ask, what is 43 by 17? Can you guys answer that? Five seconds, one two, so we don't have time to open up. Now 43 divided by 17 if you think 17 into 34, 34, and then I said 17 half is going to be 18 so. Uh. It's a 34 + 9, then you can calculate them in 34 + 9. This is 44, right? So answer should be close to.
10:33
The 2.5 ish kind of number. So that's, that's a short way of thinking that that is where we want the model to be. So we are going to hint the model and model needs to tell us immediately what it thinks. Then, we need to provide another token, which tells the model. You can think more when we provide another token that that tells the model to think high. Which means you have to have a data set for that.
10:53
So, now whenever you see Claude or chargpd or others, giving you free things that is you training, the making, the model for them. Have you came across that you are answering? An experimental feature. Yes, which answer. Which response do you like, yes, or no, that is the exact training data set that you are helping the models to create, right? So, this depth control, let's again, look at charger PD.
11:24
So this? Medium and instant. These are the pointers that go to the model where the model is told that okay, you need to. We don't have a chargpt 5.5 high or medium or. Instant, right. Those 5.5 inches might do that, but when you're coding definitely, these are the traces. In fact, codex. I don't know here. I have not here. Okay, so if you're using clot code, you will see there. You in fact have, like, full car.
11:57
I don't know what else should I call? It looks like a car to me, like the kind of features they have, I don't know. Right, yeah, full slider. This is you putting a break. This is you burning the, you see so many different levels. So this is something that. The model has to be trained with. So, again, the question goes back. Where is this data set? Who has the data set? We have a tag for think small, think less, think more, right? So that is again.
12:32
Why? I keep saying data data data data, so when you say it should be long, we also want a strong reasoning model whose depth of easing can be controlled. Simple problem should require little thinking. While difficult Proclamation should require more thinking and in Greater depth, so there needs to be the. Our process or steps stored somewhere where we have an example of model thinking deep, thinking, less thinking, hi, thinking, short, and then use the model for training. That and the answer host also has to be good or goodish.
13:02
Right, and final model must understand and genetic indic tax natively. That is the primary differentiator that we have, so it should be good in is we are basically making an Indian right. That is, all Indians are where you can speak in Hindi language, and uh, your intelligent. You can be hired by Google. Or other companies outside India, and then he can. Uh, basically settle. Right. Do you think this is how Indians work? Are we not defining an Indian? Dude, how I see most of people joining the CEOs of the companies are right? Now why the mixture is the model? This is again something, and this is the key I want to take out. This is, uh, one statement that I wanted to remember today and take it out, because in future, if you are in lab and you are training your own model. That is the definition of what the model actually is.
13:55
The easiest way to understand and make sure is the real design decision. And that decision, once you take the assetting, the DNA of the model, and you'll see the model Behavior changing. Also, as we start changing the data center as the modest getting trained. Now if more of the fixed token budget is allocated to code? And less to the generated web. The model become definitely good at programming, but it will have less World knowledge.
14:18
Right. So now, what happens is the model is really good in programming, and it can program anything that you want. But the moment it has to use common sense, that common sense is not there. Right. So now if you say. Write a model to. Counter number index finger numbers on humans? And uh, you'll be given how many humans are there in the room? If you don't have common sense, you will write a program that okay. Ask the user how many humans are there in the room and give me a list of index finger.
14:51
Of all humans because human one may have two index fingers. Human do may have a four index fingers and so on because common sense is not there. So the reasoning that the understanding that how the world works that animal has four legs. Human has two right or? The IQ of? Uh, man is 80, and woman is 400 right. All of that common sense information is not. It has to be learned on, right? You can't just like, assume that it's there somewhere so you can make a model that is really good in coding. The code won't crash with the code just doesn't work because the comments information is not there, right? So, for example, uh, book a ticket for me. You can't book a ticket for me at 2 am. It needs to be some time I can go or at least ask, so those those things that are there right. For example, write a letter to my manager that I'm not well, tomorrow, and I will not, uh, come tomorrow, so you can't expect the model to write here. I think I will not be well, tomorrow, so I will not come tomorrow. Detail some Gan and other stuff. I just needs to be there,
✅
Action Item
🎯
Decision
15:54
right? So If a fixed share deserved for indie language, the capability remains protected. So, that is also something when you understand. And today, we'll see, why in RK, especially that we need to have this additional capability preserved for index. Now, without the protection English is just going to destroy because there's so much of English content out there that we just can't match that. Now, the total token budget is fixed, so every additional step is going to be design in a way that it can give some superpowers to our model and build on top of that. So, here we have a mixture composer.
16:28
I want us to read this this mixture. The capacity decision composer baked into The Benchmark, so the data that we have here is coming from the benchmarks that you guys also selected as a part of the assignment.
✅
Action Item
🎯
Decision
16:40
This is the main pre-training run. And it is General web heavy. As you can see that here we have General web. Beta
🎯
Decision
16:46
is around 34. Because it is the most abundant data if we move the sliders here, and all you will see that things are going to change. What, but I, what I do want you to read is the last line. The scarce leans agenti can verify intake and long reasoning are held in, are held small on purpose, and the concentration concentrated later in the short annealing phase. This is not the first data set.
17:16
From a training perspective, we are talking to college level for a model. This is not a school level. The school level is going to be a lot of. Uh, your general web, so you'll start with the general web. And as we are training, and as the model is reaching that college or PhD level. This is how the data set will look like. This is, let's say you had 100 samples 34 of that will be web 12 without is going to be stem stem. Is your science technology? 16 is going to be aging two calls, then we have long context also getting in slowly. We have reasoning traces also coming in. Then, if you have code.
17:48
Right, and this is set of hundred. So if you move on, you're going to see that you have to change others. So when our model is in the college, what college do you want him to go? The more the bond is going to go in a MBA college, or it's going to go in engineering college that's going to go into, uh, arts college, and so on. And this is the behavior you are setting in.
18:09
So, as we train a model, we have a lot of data to cover through. Once we are somewhere around 70, 75, 80 data, this is how the data will look like we clear on this part. This is not the first patch. The first batch is going to be ABCD above. As a model is coming to the closure of the pre-training part, how our model is going to look like. Right now. Here we can decide the model is going to be super keen on index.
18:31
Everything is indicated. We don't have that candidate, I said, as well, but if you just do all of that, this is the this is how the mind is changing. Here it is, becoming a B2B spokesperson. Yeah, exactly, here. So, BJP spokesperson shoot Hindi and should Hindua only that logic buttons, right? And if you just train on long context and nothing on else. Now, here we have Shashid, who can answer a question in, like four thousand words, where people will not understand here. We have reasoning places right here. We have Asian Tech where it just.
19:03
Gives a command. So, here, here, if we train a model on agentic and code only, for example, and you push for code only an agentic, like, somewhere over here. Now, here, if somebody asks that? Why not putting them in the Pradhan or whatever the demands are into jail, or why you're not asking them to design the model is going to think a lot, think a lot, and think a lot and come back and say 42.
19:27
Because you've only trained on code and agentic, and it has to find out how to answer you have taken away that capacity, right? So? This is. Something that we need to be aware of, how you decide the bucket size, how you decide. What is the part of how much of the training set at that time's code or agent decreasing or freezing traces or long context will determine how good the model is. Very, very important phase. You need to think of model little like a small child, your child. You're training at an initial part. It has access to everything earlier. I at least could say that. Okay, we should think about how we I was taught. I was thought like first class second class, third class, and I thought of that will actually work also, but I'm seeing the modern generation modern kids are like taught everything they have a mobile phone, and they know every damn thing that is possible so I didn't have access to that. So that's not how I think the behavior changes. So, uh, where I'm coming from. It's like it doesn't make sense to even tell about quantum physics to or a black hole to a four-year-old and answer sounds like, yes, because today's kid can handle that they know the black hole exists, and Star Wars was there. There's a, uh, these
💡
Callout
20:35
kind of things have taken place, and Interstella and all that information, the depth of knowledge of each topic, the. So, which con, which concept introduce when is not the question? How deep shall we take that particular model in that concept? Is the question, right? That is where we need to think. Not that we'll not even talk about, uh, things that the kids are not supposed to know the model will know everything, but the depth is going to be slowly increasing. So there's something that has to be decided. This is not decided. That's why this is slider, and there's no reset. So, you need to help me part of assignment. Also, you need to think hard and tell me that you think that is going to be the right now.
🎯
Decision
21:16
I remove this data set, right? You're going to see that this also changes, so you can't think that. Okay, I am going to make a model that is super in writing. Any Indian language or conversing in any Indian language go ahead, but there is no. Data set. Right, you can't. We need around two trillion to conserve where we look at that. Right, so not possible. So there are things that are going to limit us based on.
21:43
Um, the data availability. Here we need repetition. We are already repeating our data set. Here, we have sort of covered, right? So? Uh, it's. It's not that you will just think that, uh. Want to train and you have the data set. So again, again, restriction comes in now as changes. Also, you're going to see that these benchmarks also change. That is also linked, so if you don't train on code enough, then you will not be able to Target the benchmarks. Right, so reasoning places need to be there for getting the reasoning and stem.
22:12
Long contacts need to be there for the long evals to come. But if the, if you keep all of them to 20, 20, blah, blah, blah, blah, then we have reduced our Journal Web so much that we will now feel at mlu. MMLU is General whenever you see. MML is basically a GK, uh, it's. Ips is what it's called. For example. For exams, IPS officers.
SK
Soma Korada
22:39
Civils.
💬
Adinath Auti
22:40
civil services
TA
The Admin
22:40
Yeah, mmlu's upsc. Basically, whenever you see MLE R upsc, you ask everything
💬
Sunil Jakkaraju
22:42
UPSC
TA
The Admin
22:45
every damn thing, so it's literally. That kind of data set. So this has to be designed very well, right? And now, think about it so. You and me sit together. We Define the best Transformer architecture that here the script connection will go from here to here. We'll have the modern. Optimizer, we will have. We'll use flash attention, 47 not invented yet, and we do everything possible. We write a beautiful tokenizer, and this data set is not there, and you have not trained on the data set, we forget. We just say we put in arbitrator set in a folder. We'll take our model, we ask the model. Go look at the folder and train it. You will get crap. You will get. I don't want to say, but Raul Gandhi and.
23:28
Uh, we will get things that you don't want, right? It's not that the other side is good. Everyone is gray right now. We're looking at what is happening, so it's like if I can put everyone in same bucket today. That's how things evolve, but the point is that you want someone like bhoney, for example, right, maybe yes, right? So if you want a sports person like him, or if you want some interesting people, I'm falling short of people who I can call intelligence who have not screwed up. Even Elon is then in that list, but you get the point, right? You want really, really smart, intelligent people if you want that, and this has to be decided. We have to make sure that we know what is it that we are training the model on and at what time. So, what today's session is all about, what time? Okay, we will hold for a question here now before I do.
💬
Bharath Kumar Bolla
24:09
Is this data composition a secret sauce in claude and/or codex
TA
The Admin
24:13
This is how native heavy looks like this is sort of the initial one. This is how our data set might look in the middle phases because you can see coding is also expanding, but the web is going to be a generic full context question, which is going to be going. This is how most of the annealing part of the pretending will look like and close to the annealing. It's going to become something like this so you can play around. This is my thought process. I do not have any.
24:37
Scientific backing for this. This uh widget was made using going through a lot of papers and seeing what a lot of people are doing. Right. A good way to become Sparta to see what smart people do or good way trainer models. Curriculum is to see what other people are using in the curriculum, so that is how I have built it right. So now we open for these two questions and. We have AJJ AJJ.
AJ
AJ Jain
25:01
Um, yeah, thanks. So don't, first of all, I think this visual is great, uh, very, very good to give that intuition. The question is, and probably you answer it. The calibration that you have at the right top, the capability Lanes. Uh, what is the logic that you've used to, you know, do that, as in, when you were changing one, for example, the coding name, the others were calibrating?
TA
The Admin
25:21
This is the calibration this part.
AJ
AJ Jain
25:23
So, if you select one in the mixture, it will show what the breakup should look like. Is it, for example, live code bench?
TA
The Admin
25:28
Yeah.
AJ
AJ Jain
25:30
Change this if I just select Live code bench. It will show. So, so what I'm trying to ask around is, what's the logic behind this? How have you? You know, come up with this.
TA
The Admin
25:39
Okay, yeah, yeah, let's go back. Now, what are we saying, we want a model that is accelerated coding and age intake? We want it to be able to think of long long task plan for the work call tools across multiple steps. We just recover the calls, and so on, and so on. We also want a strong reasoning model second reason, right? You can, if you ask that, what is it that my wife wants so it can think deep. It may not come back with answer, but it has a capability of at least thinking deep, what happened yesterday, and what did you forget, right? Saying that my model needs to be good, and this is okay. But how do we test it the way we test? It is by looking at those benchmarks. So, here is a coding Benchmark. Here is the agentic tool Benchmark. Here is a reasoning and stem benchmark. Here's a long context benchmark. Here's an intake Benchmark. Here's a general web.
26:26
So, when you say that you, you want a model that is good in. For example, C plus, plus, or unless a Cuda compiler or C plus plus compiler or in C plus plus,
💬
Rishikesh Kumar
26:31
Hi, I'm transcribing this call with my Tactiq AI Extension. https://tactiq.io/r/transcribing
TA
The Admin
26:35
then how do we test it then into Benchmark that is designed only for that purpose? Just list it here. Now, this is the token budget we have. And, and the sum is going to be 100 always, it's a percentage right now. If I push my model on, for example, a lot of index, then the amount of data left for to train for that particular topic is gone.
26:56
That's the context that that's the logic behind us that if you're pushing your model in a specific data. Uh, the specific Target only then. Something else is going to be lost, so they need to be balanced for that. I hope I answer. Okay. The data data Benchmark is fine data. For example, if you see.
BK
Bharath Kumar Bolla
27:25
So, is this the secret sauce behind all these clot and codex how they are differentiating themselves and also? And if that is the reason, then why is releasing models like only agency encoding tasks and some for long reasoning or something like that?
TA
The Admin
27:47
As I said, like if you just make for agent tick or agent, you can same similar thing, right? Because agent, it means that you have capacity to not only browse but scan and filter and do all of that stuff. If we just train on that right. And then you ask that. Okay, give me a list of. And I'm going to push this in such a way that we have all of them together, and that will force our general web to be as low as possible.
28:17
Looking for long context? Also, let me push good on agentic and let me push code also, really, really high, right? So, this is the model that you wanted. Now, ask a question that. Where will you find most of the vegetarian people in the world, and what percent is the population they belong to? So the model has does not even know what is vegetarian and what is non-vegetarian? Right? So, if you do not have a general knowledge, if you don't have common sense, how can you make a model? That is, it can code. The code won't crash. Code is perfect. It follows all the right right policies. Agentic tool is capacity is really good. It can call a tool and read the results and based on what the results are. I can take the next step, but the common size is not there.
28:59
You again. Have a tool you can have a model that can't be used. If the common sense is unfortunately in the web where people are discussing random things, so you can't just say that, just launch a model that is really good on only these things. It has to be a general capacity that is by AGI, right? The Jeep of G, there is General intelligence. It's not a specific specific intelligence.
29:20
Even if you think about microbiolators who's really good in doing something, he knows how to use a telescope. He knows how to drive a car. He knows how to talk to people file his taxes. Talk to his wife or husband. Right. So all of that is still required. So you can't say take that capacitor and give me a model that only does one thing that that won't be possible.
29:40
I wanted to capture this particular part. You don't have GK. You don't have common sense, then you cannot be. Uh, good programmer. You can be good coder, but good programmer or good program architect is impossible.
BK
Bharath Kumar Bolla
29:53
Got it.
TA
The Admin
29:55
For example, let us say you pose a question to a model that. I want to transmit data from point A to point B. I want to work on the best compression possible, right, without comments, and it works on it and figures out. Here's the compression mechanism I can use this algorithm. There's a one minute video I will spend one hour on it. I'll compress it to 10 KB, and then I'll send it. See, I've achieved it.
30:18
Who's going to spend 10 hours or one hour on compressing? So, that is common sense that I can't spend time on compression. Also, even that was not objective. Right. So many times, like if you're married, then you know you're given a task. But that task comes with 10 other tasks, and you cannot ignore those nine tasks. Right. Come and sit on a chair means you need to be well dressed. You need to be well tucked and everything is there. Comb and hard, okay, and everything and? And all of that has to be done. If it is Thursday, you can wear black. You may have some weird rules, so all that is common sense. You have to have a model that has common sense.
AM
Avnish Midha
30:52
You have one similar question within coding, like, how does it learn, say, designing testing? Is it like by itself, or we have a breakup within that, also of that of the training data?
TA
The Admin
31:04
We have to collect as much as possible.
AM
Avnish Midha
31:06
Okay.
TA
The Admin
31:07
Right, and, uh, again, Benchmark. Do you want your model to be really good in the website design and all this Benchmark for that? So, this list basically increases.
AM
Avnish Midha
31:17
Okay, got it. So there's a breakup within that that we can select and collect.
TA
The Admin
31:23
Perfect! The website and all are, like, such a small benchmarks that I don't think open, uh, this will even talk about it. But let's look at that type into open tool. Yes. Jill, I'm sorry. This is like killing people right now. This is, uh, this is the first time of actual threat that has come out from China. Is this model because this is competing so close, see 4.8, is already very good. 4.8 is like iPhone 12, right. I f***** up my iPhone 15 and I'm back on my 12.
32:02
I'm asking, why did I buy 115 right this? This is 12, this is, I don't know when when it will come, but this is still works, and I'm not feeling any lag or stuff and batteries also there. I'm just asking, why did I buy iPhone 15 three years ago, so this must be something like six years ago. And the dance works no scratch, nothing so. Um, I think after Opus 4.7, that's where the model are. They're already very good.
32:30
Now, the iPhone 18 will come 19 will come unless, like, I want to, like, don't feel embarrassed. My friend or our iPhone 15? I was thinking little to take a take a yellow color and color my phone orange. Sorry, so people don't be, yeah. But if you like, avoid all of that. You can see that we have Frontiers. We Post train, so SP. Marathon. All of this, you will not see Opus another model talking about.
32:55
Terminal nl2 repo D bestw tool decapsulon mCP Atlas, but this is a way for us to understand. Our model is good in that or not. This is the only way, by the way, because you ask the right question. How do I know my model can create a good website. They need to be much hard for that. I can immediately check on it and are not this. Just just push a bit of more data. It's like tuitions. You're not good in a particular subject. Go join education.
33:19
Sachin.
SB
Sachin Bharadwaj
33:26
Text, is that the phase where my model defines its maximum context length, for example, one million, or let's, say, 262 K or something like that?
TA
The Admin
33:33
Yes, uh, when we, but when we say long context, it doesn't mean that the model cannot generate 40 million tokens. What it means is that after 1 million tokens, it kind of start becoming gibberish. We don't want that the performance drops so much, the the quality drops so much that we say that. Okay, we have a 1 million it can continue, and then we have a hardware deployment. Also say we deploy a model on a hardware
SB
Sachin Bharadwaj
33:55
Yeah.
TA
The Admin
33:56
that can only handle one million in the memory, so we are again Hardware limited there.
SB
Sachin Bharadwaj
34:00
So, if somebody says that, you know, uh, these models are launched. And, let's say, the hardware supports 1 million, but it doesn't necessarily mean that, you know, 1 million, uh, will, I mean, my model might start degrading in performance. If you consume, let's say, 40 percent of that Max sequence length.
TA
The Admin
34:14
That's correct.
SB
Sachin Bharadwaj
34:16
Okay, uh? Yeah, the other question is, uh, see in this later stage of retaining phrase that we are discussing the curriculum, right? Where you want to give specialized skills to? To the pretend model. Uh, in the interview course. Uh, what is the minimum, uh? I mean, is that a heuristic on? Let's say I want to retain some skills, uh, at least at a minimum percentage level, though, that I don't lose it going further because I don't want to. I don't want to over train it too much, but let's just
TA
The Admin
34:41
Ben. Benchmark.
SB
Sachin Bharadwaj
34:43
mention. Uh, but let's say my question is a bit more on the percentage distribution, right? So, let's say, I don't want to lose the indic capability, uh. So, at least, let's say, 2 percent of it should be there based on a Max token, uh, budget. I have is a huge trick on that, for example. I mean, I'm just looking at the flooring distribution of the of different skill sets, uh, so that I don't know that.
TA
The Admin
35:05
You you? Are you are looking at a secret sauce? You will. In fact, I, I give you as a small assignment, find the research paper or a Blog, or somebody else who was speaking about this. You not even find that.
🎯
Decision
35:19
You are already looking at sort of a secret sauce, and you're saying you want heuristics? We can learn from people how much they are doing.
SB
Sachin Bharadwaj
35:26
Okay.
TA
The Admin
35:27
Not that this is. This is the secret sauce. People will not coincide it. Maybe something you will find at omo and others who
SB
Sachin Bharadwaj
35:30
Oh.
TA
The Admin
35:33
are hugging face when they train their small models? They do talk about almost trains big models. They do talk about it, but exactly what percent? Where can we replicate here? See the Restriction are different. Level first is anthropic has never released anything open. Then we have open Actually releases something open once in a while, then you have all the other free guys Google Gemini they keep releasing, but they. Sometimes it is open models. Nobody even releases data set or talk about it. Then we have olmo and a few models on hugging face that do release data set. Nobody releases the curriculum. Nobody releases the percentage share inside it. Nobody has ever done it. Nobody will do it.
SB
Sachin Bharadwaj
36:16
Okay.
TA
The Admin
36:17
You again, you will find. Go online and find one emotron. Did something like that? Yes, for a very small piece. This is a secret saucepan.
SB
Sachin Bharadwaj
36:24
Okay, thanks.
TA
The Admin
36:26
Okay, uh, shaker.
SR
shekar ramachandran
36:31
To this, but we can take it up later, too. But, uh. When we say mixture of experts right, is it also to do something with this capability lens? As a part of the training, or is it like, totally unrelated?
TA
The Admin
36:48
No, it's also completely different purpose, and we will talk about it in future sessions. It has not been proven till today that a mixture of expert model is
SR
shekar ramachandran
36:54
Correct, right?
TA
The Admin
36:55
better than a dense model. So, what, why? We are using mixture expert has to do with how poor we are.
SR
shekar ramachandran
37:04
Oh, okay, okay.
TA
The Admin
37:06
Don't have money? Uh, we will have to make a mixture expert, so a bigger mixture expert is as good as a smaller, really good, dense model. So, so it has benefits with respect to
SR
shekar ramachandran
37:14
That's fine.
TA
The Admin
37:16
cost, but it does the job basically.
SR
shekar ramachandran
37:20
Listen. Okay, because I was just curious that maybe because, as you said, I was also thinking about the cost aspect, but I was just thinking that maybe the distribution of of this capability also varies with respect to mixture of experts.
TA
The Admin
37:32
People are just pushing really, really hard right now. I'm seeing make sure Express becoming one person with the overall activate also. So earlier, it was seven percent ten percent, then dropped to seven, then four, then three. When we were training, we wait for some around three. Now, people are pushing to less than one percent.
SR
shekar ramachandran
37:47
Okay.
TA
The Admin
37:47
So, and that sounds like less amount of. Uh, with the models are bigger. They are like 2 trillion 2.4 trillion, three
💬
Adinath Auti
37:52
are most current SOTA models MoEs?
TA
The Admin
37:53
trillion parameter of that. One percent is something that's going to be 10 billion or those kind of numbers, so the active memory required is going to be very low when we deploy it. That means that I can serve more people, and if we models because the model is big, model will have a really high quality. That is what zlm 5.2 is. Right. The capacity is higher so it can meet the performance, but had the element decided to train the whole damn thing as a dense model.
38:20
They don't have data set for that, but had they had the data set, then it would have beaten Opus and other models. This is the answer.
SR
shekar ramachandran
38:28
Okay.
YR
Yashwant Ram M
38:33
Horizon tasks. Uh, what? How to make the model good at a specific thing? Like, it's, uh, it boils down to data again, or, uh. Uh, it's just expanding the reasoning traces. So, just trying to wrap my head around it?
TA
The Admin
39:05
Got the answer.
YR
Yashwant Ram M
39:06
Yes.
TA
The Admin
39:08
Now, I wanted to have a bit of historical background on this because. How all of this started all of this actually started from cursor? Corsa was one of the first few guys who were able to take a model and convert that into a coding platform. Now, moment it got converted into coding platform. You'll see that cursor cursor had a data. Where you have a user asking for a question, it writes a particular code. The code fails and then goes back to them, and it's not training. It's like literally using the model, so initial cursor is good enough, right? But that's where the dataverse data started getting generated. Now, that is what exactly.
39:47
Your clot code focused on that. I will make a cloud code kind of version I'll give. So, when you're using clot using cursor or when using Gemini using cursor, they have the data. That's where the data generators humans are. The guys who are creating this data set because we, we are the only guys who are creative, right? Today you create data sets, for example, uh, out of, for, for no specific reason. I'm making a. Application for radar cross section. You want to see that.
40:16
Do they want to see how it does cross section? The software costs around half a million dollar. But I'm making it just for fun. This is a general. Online, I can decide where the radar is. What band data is working at is monocytical or biostatic? Then I can. Run the I will not refine it to take all my GPU when I'll run RCS RCS editor cross-section scam right now. This is the cross section.
💬
Adinath Auti
40:59
Do the B2 Spirit RCS please !
TA
The Admin
41:02
Where I? But this is not what people want. People want Globe. And is running on parallel GPU, so it's consuming all of my gpus as much as possible. Politicus. Take some time right now 9 seconds and. 2100 aspects. It will take some time, so when it comes, you'll see that there's a 3D block. So, if someone wants to make a jet using the materialam.node here and install it as at that location. Then I have applications.
41:33
I don't know how many people will think about this, but because I forced Claude code to think about this cloud code, had to go online. It had to look at some sources that I figure out how to optimize all of that data. Yeah, this is the blog I'm talking about. And this reflection of all and the first one. I don't know why anyone would want to. Uh, see that if the radar was there, then how the reflection of radar Reflections will flow, but I wanted to see it?
💬
Abinesh Sankar
42:05
+1 B2 Please :)
TA
The Admin
42:06
Now, that's where that is. How the data set is created, right? This is low pattern, and yeah, this is exactly how the radar. Cross signature for this aircraft will look like. Right, then I also went and made this other application, which is.
42:50
Let me select the same plane again because it looks really, really cool.
💬
ADITYA SINGH
43:04
Hi, I'm transcribing this call with my Tactiq AI Extension. https://tactiq.io/r/transcribing
TA
The Admin
43:09
Conditions for colors, material. It has look cool also, right? And Increases need to be smooth, so all that crap. Then, I would like to see some flow lines how the airflow will look like around this particular chat. Then, I thought, why not put a GPU based particle accelerator. Also, it's not showing up right now. Probably something. Yeah, it's there. And then I want to control the density. I wrote a smart algorithm. I have four slot code to write a smart algorithm, the only those areas that are important.
💡
Callout
43:40
Are there?
💡
Callout
43:42
Then capacity, then they can be thick. Also, they can be thin also, and then density can be increased. The visualization is cool right now. I know exactly how? The and it's a proper jet engine, by the way. Then, I thought, okay, if, uh, that is there, then why don't I make? Jet engine designing software itself, right? So, this guy actually designs the data and a still work in progress. But it's not, I'm not, I'm showing off. I am showing off, but the point is that using all of this, the data now goes to plot
💬
Suresh Mantha
44:11
all these params can't be linear... they need to be random combinations
TA
The Admin
44:13
core and Cloud code. See that this is how particular problem is solved, right? And if the problem took, let's say. A hundred thousand tokens to solve. That becomes the long context information. So, where are the reason prices there? Here are the heating traces are, and that is basically the reason why we need to focus on a tool that is really good at coding and agentic, because then we can hope one day to launch something like Cloud code, which is cheaper. That's why people will use it but cheaper than, but it allows us to collect a data set.
44:46
Right. The reason? Xai or Elon bought cursor. Is this they are sitting on amazing number of users, and if they can move them to? Uh, their model, and they have they're setting up literally at a lot of people who will start generating data for that data for them, and then they can train them all better, and that that's the. That's the. Sickle cells, basically, which is publicly known, but indigenous fails to implement it. Okay, check out last question, then we move.
CT
chirag Tagadiya
45:19
So, you said, for example, if companies are advertising, then they can support 1 million context window. So, does that mean, uh, is then? The meaning is like, after one million context, the model start degrading, uh, its performance right performance will start degrading, or I think some you mentioned that some. I think Sachin mentioned that after 40, also it can reduce so. Just want to understand what, what does that mean if if companies are saying that they can support 1 million context window?
TA
The Admin
45:51
On General around a million tokens. The model maintains the forms. Let me show you.
💬
Max
46:08
context limit is absolute: if 1M then it can't exceed that
TA
The Admin
46:09
Yes. Seem roughly. You see that. So nearly all the models gpto shows that okay, average correct correctness is occasionally. But as the context is increasing, you'll see that models the weaker models they start dropping. And here it is shown to 125k, but if you push it to 1 billion, it will be stored stabilish around 1 million. After that will start dropping, and this is the average correctness. This is not for.
46:40
It is not touching one, which means that if you ask, what is four into four one million times, it will give a good answer, their performance going to be one. But if the question was very complex?
💬
Max
46:49
for < 1M: see "Lost in the middle' paper
TA
The Admin
46:49
And a lot of thought has already gone in the complexity in the initial 40K token that will not hit 1 million, so you need the first point second point. If you
CT
chirag Tagadiya
46:54
Okay.
TA
The Admin
46:57
design a model which claims that it has a 1 million context, you will not provide it. A hardware that can go up to 2 million also, right?
CT
chirag Tagadiya
47:04
Yeah.
TA
The Admin
47:05
Because then you have to commit to a bigger RAM for each user, so that is also limited. So when we say 1 billion, there are two crops. One crop is that the hardware is restricted. And second is that there is average performance that the company is. To provide for that, that unit that that unit of context.
CT
chirag Tagadiya
47:23
Thank you! Thank you for that.
TA
The Admin
47:26
Okay, codex reduce it. Context length, maybe because of that, only they were seeing some average user user how much it's using. Okay, now let's move to that. So this is whatever I spoke to you written in text, so you can read if you're not in the class. Now, composing backward from The Benchmark, right? This is. Uh, what we need to do and sort of. This is something that.
47:52
Is also required from parenting point of view. If you think right, what is it that you want your child to become? It's a very controversial question to ask because he can become anything okay, but you should not become stupid. You should have some other skills also. So, for those kind of things, a lot of parents also think backward, right? Get a really bad math. Okay, we don't want him to become, let's say, a superb at Olympiad expert, but some expertise needs to be there so you can actually pay his taxes and so on.
48:20
48:55
you will see some of your your own listed. So, what is swe bench now? SWV bench
🎯
Decision
49:01
is a human validated real GitHub issues paid with the actual repository. The agent must write a patch for. Very interesting, right? It's a problem on GitHub, then a human batched it. So there's a solution also, and then we have. Yes, it was solved, so that is SV, SW, Benchmark Ripper level, bug fixing. Now, we're getting the real code base localizing the fault, editing the code that makes a hidden test pass metric is whether it's resolved or not, right, so? This is how the data is going to look like. Now, green is supervised in the
✅
Action Item
49:34
loss, basically part of the loss. Gray means masked context or observation, and wallet is a reward. So now, when this data is trained on, the issue is not trained on. Issue is not part of the loss. You should be clear now. That is why I spent some time in the beginning that some part of the data set, a particular example. We will calculate loss, and we will not calculate loss. So, where the GitHub is saying that? Okay, our code is saying, I have an issue you will not. You know, you will not ask the model to predict where the even before it looks at the code, we'll ask. Okay, look at the repo name, and uh, tell me the bug. That is
✅
Action Item
50:12
what means by training the? A training on it, right? So, repo fires has not renounced a reader, so it has derived everything. Okay, now the green is the calls that were actually made by the human or the plot code. When China is healing, or when we steal, or what are the patches that were signed? And this is how the assistant pass was signed. You can see that it called a tool. It removed this line and added this line and this line. Can you see that this is how coding works. This is what clot code is doing behind when you're using it. It looks the file looks at a part of the file and says, remove that. Add that.
50:45
Okay, then test was run again, not trained on. Now, in the verifier scores that okay, the the bug was solved, then we rewarded reward, something we have not discussed yet. We've only discussed loss soon. We are going to discuss reward. Also, if there's a loss, there's a reward as well. Now that is just SW bench, which means that if any one of you get this data set to clean and get this data set to work on. This is what is expected, right? I couldn't explain this to era V4, and we started collecting data set. It downloaded the whole thing.
51:17
Put it inside a folder, tokenize it, clean it, normalize it the way they want it, and put it inside a folder. And we are training on it. So in V4, we train on all of this. Which we can learn what the code base should look like. We train on the wrong code, part of the code base, and then we're asking to fix. Now we have SWA bench Live Pro.
51:37
This is a fresher, newer data set, contamination resistant, harder variant of swe. So, SW bench verified is 100 right now. In some benchmarks, so people have cracked it. So now we have bench life Pro we have, so we have Pro. Then we have Pro Max, then you have Pro Max 15, then you will have Pro Max plus plus, and so on. Then you have Ultra also right, then we have terminal bench task that must be completed inside the real terminal initial reasoning. So, only inside a shell again, you can see, install, and configure x18080 was a task given. Now, what is a model train on, install the package and change the listening to the port configuration? So, this is the model thinking Trace.
52:13
This is the command that the model generates. We train on it. The shell output is something like this. If the problem was solved with a water model if it was not solved, we punish the model, but I hope you are getting the idea. Now, we have tall bench. Tool agent, user interaction, retail, and Airline, where the agent follows a written policy across many turns. So, here, again, cancel my flight booked. All this is trained.
52:33
True return is not trained. Then we train on in this and the second one. Then we have BFC here. Berkeley function calling leaderboard. Similarly, we have web Arena, say hosted websites and Enterprise workflow. The end the agent must operate purely web browsing. Then, we have got a general. As I said, question needing tools, browsing, raising, and easy for humans hardware model. Now, this is what makes a model a really, really good chat bot and really useful.
52:56
Then we have browser comp, which is hard verifiable web browsing for hard, hard to locate, Etc, right. So, these are some of the benchmarks, but I think we need to be targeting. Then live code benched and AI co-pilot core forces aim, which is American International mathematics exam, literally Olympia level, and your Frontier math research level mathematics, and you can again see that what part we train on what we don't, right? I spend good amount of time on this particular one, because this needs to be clear today when we send a data milus Bharat. And data set, and then we have intake and data set from. Right. So here you can already see we have. So that kind of conversation is happening.
53:44
That is, how data looks like here. You have prompt here. Your expected answer, right? So, depending on the data set, depending on the Benchmark, we need to look at what The Benchmark is. Then we need to find tools or find data set that can solve that Benchmark. We can't read on. Benchmark right, Benchmark is set of like thousand questions. You can learn the following question, but you don't have you have not understood the capability, so you have to find a bench. Find a data set that helps us solve that Benchmark, which is what we were talking about here.
54:16
If you're targeting Live code, and as I said here, I'm just moving the coding thing. But inside, I hope, now you understand what is, what is this code earlier, which is render on downloaded all the GPU kernels and thought we have code right now? I think you should get some idea when we say code. What is the meaning of code? Are we clear? Any question on this? Nikhil.
NS
Nikhil Shrimali
54:49
Tools that can solve that minimum, like, are you talking about? Let's say, if you are like solving coding benchmarks like we would need like? Terminal session, where we can execute whatever has been generated, and then it would be like the result of that would be used.
TA
The Admin
55:04
Yeah, the model has to have the live environment to and that infrastructure we actually didn't build also. In the in the V4, so I can't, I should still blame, but that infrastructure we didn't build while training the model. The model should have access to a terminal bench and actually launch a terminal and run commands there. We never had that. So, I hope you are getting where always going. You just don't have a model and data set and train and loss. Now you have a computer computers where you can launch a terminal, write some program, and get the feedback actual feedback.
NS
Nikhil Shrimali
55:38
Okay, and for others, like, for Weber net, would be web access, and browser could be browser okay.
TA
The Admin
55:43
The moment you say, tool call moment. We have the tool calling things here. Moment you say top bench, right? This means it has to have, where otherwise, where will it run this command? It's not similated, right? It can generate some random command. We need to run that command and then give the error. Let's say data spelling mistake. We are not fixing to con. We are the only way we can train the model by giving the feedback do this happen.
56:09
And we can't train on it. You can't train on this. It needs to fix. This is the only only signal only point where the reward goes.
NS
Nikhil Shrimali
56:17
So, for a coding assignment, are you matching on like the code that is written?
🎯
Decision
56:23
Or maybe the result or the output?
TA
The Admin
56:26
Result, result, result. And sort of. It's linked with the code that is required because there are sometimes only few ways you can do it.
NS
Nikhil Shrimali
56:35
But are there patterns when model hack the reward function in a wrong way, which we recently seen, like the open air modules fact hugging phase to cheat on the Benchmark, right, open gym Benchmark? So, how do we prevent that to happen? Like, do we have like an intermediate checking as well that?
TA
The Admin
57:00
A philosophical question we can discuss at the end of the session. There is more more marketing. There's a very good article on guardian that. It's very similar to what anthropic wanted to do is more on the marketing side.
NS
Nikhil Shrimali
57:07
Okay, but like?
TA
The Admin
57:07
It's not a model problem, so not even in the context. It's a agentic clot code
NS
Nikhil Shrimali
57:10
Sorry, but do we have like them? Check that, uh, if let's?
TA
The Admin
57:18
problem. This is not Opus problem. I hope you understand.
NS
Nikhil Shrimali
57:24
Okay, so like, let's say, if you want a model to Output 42 by writing a code, but it's just like saying, print 42, and then it's giving the answer return 42. So, do we have checks like that? Also, that, you know, model is not cheating, and it is actually doing the tasks.
TA
The Admin
57:43
Yeah, we have to. We have. Okay.
SK
Soma Korada
57:51
Data sets. It is we, who are defining what part of it needs to be trained and what part needs to be ignored, right?
TA
The Admin
58:00
Yes, open and anthropic are not. Doing it for us.
SK
Soma Korada
58:04
Yeah, so that means it is to your discretion that you're saying the tool return doesn't make sense to you.
TA
The Admin
58:13
It doesn't make sense to open a and Claude plot also.
SK
Soma Korada
58:16
No, no, that's true, but I'm saying, like, we are defining that, uh, so we'll have to look at each data set and see what they actually have in it and then
TA
The Admin
58:21
Yes.
SK
Soma Korada
58:27
decide what part of it we do not want to include for training.
TA
The Admin
58:30
That's correct.
SK
Soma Korada
58:30
Okay.
TA
The Admin
58:31
Fortunately, 90% data is already segregated, but we still need to validate. It may be.
SK
Soma Korada
58:35
Got it. Thank you.
PD
Pranabesh Dash
58:41
Can be copied and be present at multiple places, so uh, as this company is taking enough for care just to discard that okay with a deduplication or something that that is not getting into the training, uh, the responsible to make sure, or like, if they trained, it can be found out that they trained with that Benchmark.
TA
The Admin
59:02
100%, they need to deduplicate. Second, there are tests that the okay, when we discuss benchmarking, I can take this opportunity also. And when we go deeper into benchmarks, we'll discuss more of it. So please don't have a follow-up question on this. Benchmark generally has three parts. One is called train. The sample examples that you can actually train on. Then, you have validate. Sample shared with you for your test, so you know? What is the percentage? Let's say you get 86 percent and then test. The results are never shared. The solutions for this is never shared. You have to take a model.
59:53
Send it to them to an API or something. They are going to run the test, and then they are going to share the results with you. So, if you cheated here because you memorize it, it became 100%. Here, you would still get 84 percent. This is publishable. I hope you get the idea. This can still be had at least test is available. Send it to people he can solve it. Get the answers 100 and then train the model on that. So it's not that we can't cheat it, but no, the new Benchmark. They keep changing this, just like, neat. All right. Okay, let's move on, so I hope you did the idea. It's not just about.
01:00:38
Collecting data and training, the slot of work that goes inside the moment you see a benchmark. You should do this. Oh, now we need to make sure it's part of data set, and that's why this cannot be decided at the end of the end of the training right end of training training. Okay, knowledge, decide. What do we do, right? We are. We are in the. Hisar in haryana or some other location, then we are saying, okay, let's make a nuclear power plant here, or let's make a semiconductor.
01:01:03
Here, where there's no electricity, no Road, nothing right. So, you need to be prepared of what you want to do section.
SB
Sachin Bharadwaj
01:01:13
Are like a reinforcement Loop here. Okay.
TA
The Admin
01:01:19
20 what is their losses? Not there.
S
Sujay
01:01:22
When training, are we looking to minimize the loss or maximize the reward? They compliment each other. Is it like if the loss is less? The reward is more, and if the reward is less losses more, is it like that?
TA
The Admin
01:01:39
No rewarded the sum of all the losses. So, maybe one, plus two, minus all the losses, something like that. So, we will talk about it when we are discussing.
S
Sujay
01:01:47
Okay.
TA
The Admin
01:01:50
Okay, so what actually exists to train on right now? We've discussed that, okay, this is what we need, uh, that is what we need, but we need a shopping list, right? Because if you don't have data to go out and look for, that becomes the problem. Now, second thing that I wanted to understand is. Sizing has to happen in two categories. One is going to be number samples. Now, there's going to be number of tokens. You can say that I have 10 million examples, but each example 10 token doesn't give us anything, so we have to have a good balance of both right. For example, I ask you that, can you guys share all of your clot code data or a cursor data with me and we'll train on it? That may allow us to get a maybe all of you have thousand examples, four months, four thousand examples, and 100 students. So, it sounds like a good 400 000 examples, but each example is only 20 000 token. The quality is very bad, right? So the so whatever we're discussing has to have two things one, the number of samples, the other number of tokens for each sample.
01:02:46
Write them. Measure different things. Number of samples are talking about sort of variety of what we have. Number token is talking about how deep that particular context has gone, and both of them fill a different kind of bucket variety definitely we need. And, but we need depth also, but we sometimes don't need depth. So, because we have that slider, right? We are thinking Ultra or medium or hard or other stuff. So that is something that is required now. This widget, which I'll not take a lot of time here. I will like you to spend on. It talks about every single data set that we need to look at for training our model. So, when we go on each of them, for example, we talk about.
01:03:25
Estaquito stark V2 has 600 million samples, right approximately around 900 billion tokens. So doing the same thing. This is big, good enough, but you'll see some other benchmarks change drastically. So, let's look at tool bench tool bench 120, 000 examples, but only 80 million tokens. So you can see that each sample is very small and Apache, so it's good tool bench, a multi-tool instruction data set over the real rest apis. So, here it can become really good in making websites and web services.
01:03:57
What is this bolt-on the bread of real API composition, a multiple chain samples and tokens and? Running per slot token, so we have code of these many. Then we have agent degree reasoning, so this depends on what is that we want to do, right? Now, if you go to stack V2 stack V2, 600 examples, 900 billion tokens, that's a big one. 900 billion tokens will close to a trillion, which probably will be 25 of our own data sets. How much of this should we take? Bitcoin duplication or duplicated permissive source code copies from the standard Heritage anchors law real repository across 100 languages. So, there, we need to go and see that. Do we need to support foreground a language that is there, so the cleaning has to happen there also? Hey, would you want to support a model that can code in Fortran? I don't know.
01:04:45
Right? So Rebel 5 can entry, and so on. So this is something that I wanted to
💬
Adinath Auti
01:04:45
yes
TA
The Admin
01:04:49
spend some time on to understand how each of the data set looks like, uh, what are the samples inside and how many tokens are there per each sample? And the many. For stem, we have dclm, then we have fine. These are big ones. Right, dclm is a really good data set. Then, we have fine web. Then, we have D2 and good good long examples. Okay. Now the training stages and where the reasoning enter. So we're talking about the coding examples, the agentic tool calls, and the normal web data and other stuff, but the model slowly is going to move from random waves to pre-training.
✅
Action Item
01:05:27
That has happened because we train on an initial big bucket of web data. Then we have the any link stage, any link stages where we are fine-tuning the proportions. What we discussing today is any lean stage. We're not even discussing sftr, our preference, training, or reinforcement. I want you to have a good idea last one, two, three, four sessions. This only today's session is this, only then we have supervised fine tuning, preference, training, and reinforcement learning coming in future. Okay, have a very good clarity. We are somewhere in between this and this.
✅
Action Item
01:05:57
Okay, the engine reasoning slot only makes sense. Once we know which stage is going to use them, and what model is expected to produce. While we are learning that signal, so this is how the training cycle will look like. So, pre-training is 95 of all the data. That's the good part. So, let us say we have a four trillion token design. For training a chunk of it. 3.6 till in tokens are just the pre-training. Then we have mid stage or an healing part, right? This is where we have around to.
01:06:27
I'll fix the visual. I thought I fixed it. This is where we have just two percent of tropons. This is where the objectives become the short learning cooldown. I'll talk about that. But here, the high quality document comes in. We have still the cross entropy, same kind of loss. Here, a very, very high quality of documents are saved, cleaned special, where these are like PSD level topics. Then, we have sft. Sft is where the agent tick traces and the chat bot and the user interaction that you saw starts the data set that we discussed earlier, coding another feedbacks, not write the code, fix the code kind of problem comes here.
01:07:06
Then the stage 4 reasoning traces this. This is where we have the long answer short answer examples, given that if someone asks that question and says that think faster, then this is the answer. If you say, I think deep, then this is answer. Think as much as you can, this is the answer. All that training data goes in this stage, then we have preference alignment. This, we are not discussing that today. This is where the actual RL and other things will come in in preference alignment, uh, we align with what the government want us to say or what the society wants the model to say and not say, right. Those are the things that are going to come here.
01:07:37
So, stage one we discussed earlier today. We are discussing stage 2 and C3 stage four one, two, three, till here we are still on the normal cross entropy kind of loss. We have not gotten deeper into RL, but we will start seeing grpo and these
💬
Suresh Mantha
01:07:47
Anything about model security?
TA
The Admin
01:07:50
algorithms that come in the Deep RL part, uh, basically releasing stress. This is where the Deep Arrow part will start also. But today, our focus is on stage two and phase three.
💬
Suresh Mantha
01:08:03
And ethics
TA
The Admin
01:08:05
Yeah, let's look at each example prompt also so, uh? Here with this training, so the loss is on every single green thing. The theory of late tectonics explain how Earth outer shell breaks into moving plates that carry the contents across the globe. When we move to the mid-size training, high quality in the valid proof, each of the followers from the last of the stated archive documents and good legal documents and good, uh. Delete the liquid data, documents, and other stuff.
01:08:32
Then sft style. We now have prompt user, explain why the sky is blue, not loss, is not on that losses on this assistance. Sunlight cutters, blah blah, blah. Then, we move to the next one. These increases again. This is not where the losses loss is here, right? And if the answer was correct, then a reward is given. So, bit of RL starts coming in. Preference alignment. We give many examples there again. We have something and something and something, so we reject some answers and we allow for some answers and serving is literally. Serving on the internet. To people. There is no loss at that stage.
01:09:08
Okay, now the agentic slot. The Aging data is an is the newest and the least familiar capability slot. This is still coming up. There are not enough papers also on this particular topic that. How do we train a model to be good agentic model to begin with, so harness can actually utilize it before. Had none of it, we were not ready to even look at that particular stage. We were just looking at pre-training and making the pipeline ready so we can train that model also. Now, imagine the user asking, uh, the model that.
01:09:31
Can you find all the research topics for a particular topic in the United States for a particular field that I'm working on so I can and find their email addresses. Uh, so I can actually email them about the services that I offer, that I know these companies or these universities might use. Very complex, right, and all that? There's no single tool that can answer this. It has to do a lot of search. It has to understand what you are doing. Maybe go through your PDFs. A PDF reader has to be there. Data comes in in a weird format, cleaned it up again. A tool is required, then read. After reading, then launch an agent that agent or launch, or or tell itself, you can only go online. Search these. How do we search again? An agency call a function call. I need to know what functions are there.
01:10:17
Then, or what tools are available? Get the tools get the result back result, look bad. Screen that multi-step, right? So this will include the planning of the step, then searching for the grants reading the results. Launch follows searches based on what it finds recover a source if missing. Try another out maintaining a standard understanding of the entire investigation. Right and do it? So, a long trajectory contains decisions, tool calls, observations, failure recoveries, and updated plans.
01:10:45
And all that is with you with me. The guys who are on cloud code. So, the more we use clot code, or the more we use codecs better, it's going to become, and you can see that now. You're going to see chargpt and charging has caught up finally after codex. If you think about it where anthropist started beating them right, but now they're catching up because codex is there, and they're providing good limits for people and one of the best thing about codex is that if you ask a question. It was 100, doesn't stop, it finishes that, and because they want to collect data.
01:11:17
Okay, so here we have a long trajectory, uh, long to the tree is going to look something like this user. Find the same question I have mentioned. Then assistant is going to plan, then we'll have the assistant tool call. Then we'll have the tool observation. Then, after that we have assistant and reasoning call. Does it look like your plot code now? Audio cursor. Right, and this is one shot function user answer, but this is exactly what you are turning on.
01:11:56
Right, you ask your question? Bill me this assignment? It does. Oh, this is what you want. Yeah, I go ahead, go ahead, switch things and things, and things and things. And oh, b******* you did not that did not that did not that I want the color change, the UI chain, etc, etc, all of that. Assignment is basically. Okay.
🎯
Decision
PD
Pranabesh Dash
01:12:18
Person was collecting data. They were routing the queries to all the standard llms. So, did they train anything so that their response to who was better with time or now? Xai will use their data and then get better.
TA
The Admin
01:12:33
All right, they didn't have enough money to train, and the objective was not to train. But giving that data to Gemini, you were giving that data to Cloud code you were
PD
Pranabesh Dash
01:12:37
Okay.
TA
The Admin
01:12:40
giving that data. Opening area.
PD
Pranabesh Dash
01:12:44
Okay, got it.
TA
The Admin
01:12:47
All right, so that's how the energy transfer cortex actually looks like. Now the reasoning effort. Now, this is again same thing that again data is required for that. Now, look at this inclusion exclusion floor 100 plus flow plus something, something 25 2.7. And here, problem is, how many integer between one and thousand are divisible by three of five. The correct answer is 467. Then we have the medium traces, the high traces, and then the ultra traces.
01:13:15
This needs to be built. This is not freely available, and you can see for each of the level the number of tokens required to enter that. Okay, the answer is now here. I couldn't show you one example where? You'll think that okay if, for example, only take five token. Why should I take five thousand tokens? That's not the logic here, but logic is that we have some questions which require a really, really last. Large amount of steps to think and then approach that kind of solution.
01:13:47
Uh, opening hours to solve something right? Yes. So? Uh, this one. Now here, if you see how long this would have taken, solve a math problem. Friday Friday Friday of 1917. Undimensional. Dimension, but I think it worked for some few hours, and that are long thinking. Ah, here it should be there. Block, test, and compute. No, not this one. I read somewhere. I don't know where it is, but I read it worked for.
01:15:01
Hours a few days, so that's the long. And this is one example. Now, they have them for themselves who took for the train on right, so releasing effect effort. We need questions that need long answers. We need question that we need short answers and. Then allow the model to retain on it, and you will train with a tag. You will train that question. Tag tag is low, medium, higher Ultra, and then expect the model to give answers after some X number of tokens. It's not that we say that only take thousand tokens. Sometimes low might mean 10 tokens or say thousand tokens, so you need to set that. Also, that what is low end? What is the boundary flow? What is the boundary medium? What is the boundary of high and Ultra? Uh, command.
KS
kamran shaik
01:15:48
For agentic task, the some of the tool responses can be quite huge, and they
TA
The Admin
01:15:53
We've been discussing that we don't train on it.
KS
kamran shaik
01:15:53
itself. No, we don't train on it, but then we have to feed those as part of the context, right? So the context there is a chance that the context might blow up.
TA
The Admin
01:16:04
That's correct. The chances are the chances the chance is always there. What's the question?
KS
kamran shaik
01:16:16
Ah, so. How do we make sure in these trading tasks that we are not?
TA
The Admin
01:16:21
No, we don't make sure we train for it. You're saying that, please save me. What if my child Falls and get injured? No, he has to fall and get injured. She has to fall from a house and get in and broke a leg and all that has to happen for the model to be careful. It is going to be deployed in real world, right? And so it has to happen. You just don't punish for it. Don't punish the model to read a large log.
KS
kamran shaik
01:16:49
Okay.
TA
The Admin
01:16:50
And again, this question more more on the harness point of view. If your hardness gives a full 4000 line error instead of just finding telling their 4000 line of error, was there, right? But this is the actual thing, then your model is going to suffer. So, the question you're asking from the harmless point of view? Model should be capable.
💬
Shias Abdul
01:17:08
Hi, I'm transcribing this call with my Tactiq AI Extension. https://tactiq.io/r/transcribing
TA
The Admin
01:17:08
Sachin.
SB
Sachin Bharadwaj
01:17:14
Why can't model autonomously decided I need to have a higher photo of reasoning and I have to have a lower?
TA
The Admin
01:17:17
Because you always reward for high. You because your reward for? The correct answer? Because the reward for correct answer model will decide I'll take more time. It's an employee you give it salary. So, and you're saying that your bonus is going to be amount of hours you spend, or if you give me the correct answer. Right, and you give 40 problems, so we'll keep on solving even if the problem can be solved in one minute. It will take full day.
SB
Sachin Bharadwaj
01:17:41
So it's not efficient. I mean, if we set it up that right? Let's says bias towards a lot of reasoning, and if I give a simple question, then it will generate a lot of recent tokens right unnecessary.
TA
The Admin
01:17:54
Yeah, because it has a have an objective, right?
SB
Sachin Bharadwaj
01:17:58
Yeah, yeah, all right. I got that. Yeah, thanks.
TA
The Admin
01:18:02
Okay, now selecting the best data while the Run is happening. So right, now, what we discuss is that? We have data. We have selected the proportions, but that is not enough. We need to select the most useful data continuously, and this is very, very important.
💡
Callout
01:18:18
Right? Uh, what did the model already knows what you're turning it on because they we do not have any control on that? For example, if I tell you now that you need to know, really, really like, you need to remember that four plus four is equal to eight. Now, at your age, you're going to say, what is this? Why should I even we wasted the compute there? Right. So, at any point in training, we need to make sure the data that is going to the model is of high quality.
01:18:43
And that problem was solved by this paper called Opus, and we used this and I could really see big differentiator. I'm not sure we're going to be using this again or not, but that is going to be a question of the overall. Uh, budget. We have to train. I'm for it. You will also go through it, and when we discuss open Opus is much more detail, then you will. Then, we can actually have a more conversation, but I want to explain what Opus actually is. Before showing you any of this. I will draw some things because. Okay.
01:19:21
So? This is your model. We will always maintain a copy of this model here. Exact copy, right? So if this model is trained for 10 million steps, we just have another model here. 10 million steps, exact copy, memory copy literally, not, not a special model, exactly the same model you clear on this. Just keep showing me a thumbs up. So, we have a copy of the model in the same GPU for short time. So, realistically, we are going to be training this model, for example, from here to here and stop in between and load this model for here to here, then stop. And then go back to training a model and then stop, and then that is what is happening. Okay, so both are exactly the same modules.
01:20:06
This is the actual model that is being trained on. What we are going to do is let's say we want to Target these benchmarks. We have Benchmark one, two, three, four, five, six. You'll have Benchmark questions itself. For example, right, we are going to create a data set that is literally The Benchmark. This is a exact copy of windfall. You'll understand why this is not cheating. We take that copy of The Benchmark. We've send it in a model and we'll see how bad that model is. You're going to get some loss, right? We're going to get.
01:20:39
It's really bad because we have not even trained on this Benchmark. We are in training, but for these benchmarks, we'll know. The loss is very high. We're not interested in the loss we are interested in inside the model. The model looks like this kind of neurons or this kind of structure. We have these parameters. We're going to see. That to get that loss basically resulted in a high weight update for this.
01:21:04
When we calculate the loss? We calculate the backdrop. We don't update the model. Update the model stealing. We back prop and we see which weights are really bad. Which weights need more support you clear on this? We take the model, the one we are training on. We take the Benchmark, we run the model on The Benchmark, calculate the loss back propagate and see which weights are really bad for us, which way it's really, really need a lot of support.
01:21:31
Right. This is called a ghost model. We're going to discuss these goals. Golden proxy this is called golden proxy, the. Uh, proxy or golden data set that is telling us how bad the model is. Then, let us say we take one to two four samples. Uh, we were about to train on the model. We take the model same model this model. This model doesn't matter. But generally, we train for 8K sequences of 4K sequences or 16k tokens, right. Examples are wrong. Let's say one zero two, four samples. Each of 10 000 tokens, for example. That is what we are about to train on what we'll do.
01:22:07
We will take all of them, but will not take 10,000. We only take 512 very short part of the example. So now, I'm going to take 1024 and I'm going to take 512 tokens and I'm going to send it to the model. And for each of the example, I'm going to see how much that example is affecting these weights. If the sample number one affected these weights a lot? I'm going to keep it.
01:22:31
In the sample number two did not affect these weights. In fact, we're affecting the weights that I don't want. I'm not going to keep it. So, we filter the examples that allow us to affect the weights, which are a big issue for getting good on The Benchmark. That is, what is opens? Right? So Opus is essentially.
✅
Action Item
01:22:53
Figure out that a sample that comes in. How much of that is affecting my Benchmark and literally? Throwing away a data set that is not required for us. Now, here we can keep, what is the keeper fraction? So, let me put it to 50, 50, basically means 100 samples, given we are keeping 50, the top 50, which affected the model in a good way and not the rest. We can drop it down to big number.
01:23:17
Also 15, we're only keeping 15s 100 percent only 15 by use. Now that will, that will determine how much of the? How much of this is? I've been getting affected because your data sets away, or you're throwing samples away, then you might be. Ah, basically getting locked here. Try to generally. This is assuming that the whole data set is good, but the good part is that if the data set was bad, it was only slash B, slash B, slash B, and some random stuff has entered. Our example open success is going to throw. It is going to say I'm not learning anything from this. My weights are not getting affected. I don't want the sample, so a really, really good way of real time filtering.
01:23:55
Now, this is done real time because, as the model improves, it's going to learn four plus four equal to 8. Why do I need to know this? I don't want to drain on it. Right, and it will quantum physics and this. So I've never understood this. It is affecting my weights a lot, so I'm going to keep it. So that is what we essentially do here now.
01:24:11
Here's a balanced one where we have English reasoning and sobering capacity, but depending on how we are. Changing the fraction of the data set that we need to keep. We'll start throwing some of the examples. That means that Opus can. And depends on Benchmark, right? If I keep the benchmarks heavy for code code index code example, so in the one Benchmark of the agentic part was not there? In the uppers, then we'll throw the internet example.
01:24:36
So, that is why, and most of the benchmarks are on the English side and coding side, so they will throw away the indic. So that is why we had always on intake so indic and agent tick always on because agentic text looks like a log log is not a high quality data, so Opus will just throw it so we will preserve it. And then we'll decide how much of the data we want to keep.
01:24:58
This is again our slider. This again slider here, but in real life, we need to figure out how much of the samples we're going to keep and how much of it that you're going to throw away, right? So one of the really good example of filtering the data at the? Uh, as the model is getting trained, any questions? We will not hit the weight.
CT
chirag Tagadiya
01:25:29
You said, uh, you will not do the back propagation. Right? Okay.
TA
The Admin
01:25:35
Back progressions required.
CT
chirag Tagadiya
01:25:37
Okay, so when you if you don't update when you do the back propagation, so you know the Delta, like which weights are changing very frequently, or like
TA
The Admin
01:25:46
Correct?
CT
chirag Tagadiya
01:25:46
changing the or the magnitude is higher. What do you look for?
TA
The Admin
01:25:52
That still doesn't matter. Data quality is very high, has no purpose. If it is
CT
chirag Tagadiya
01:25:54
Okay. Thank you.
AM
Avnish Midha
01:25:59
Yeah, what if the data quality itself is better than the Benchmark quality? I was just wondering.
TA
The Admin
01:26:09
not meeting a benchmark, a very high quality data on Sanskrit and Vedic has no relationship with the benchmarks. We are targeting. The question is that what part of this data affects The Benchmark? That is, the only question opens is solving.
AM
Avnish Midha
01:26:23
Okay, got it.
TA
The Admin
01:26:26
Nickel.
NS
Nikhil Shrimali
01:26:28
Uh, hello, you can hear me.
TA
The Admin
01:26:29
Yeah.
NS
Nikhil Shrimali
01:26:30
Okay, so Rowan, like, why are Chinese models are, you know, coming at par with the with the US models? Is there like the data quality. They have improved a lot, and then it's closer to what we have globally.
TA
The Admin
01:26:45
No, because they keep distilling it and anthropic analysis. Keep crying about it. Right, and topic doesn't have any other entropy to steal from. But Alibaba and all Chinese guys have anthropic and opening to steal from. That's why you'll see the this. You can see, there's always a slight Delta.
NS
Nikhil Shrimali
01:27:14
Okay, so essentially, the quality of data is getting as they are distilling.
TA
The Admin
01:27:16
I saw this is the one. All right, and the Delta will keep on reducing. And this proves why, because everything is there.
NS
Nikhil Shrimali
01:27:32
Okay.
TA
The Admin
01:27:35
Sachin.
SB
Sachin Bharadwaj
01:27:40
Because I think Opus will have access to the agent Benchmark right genetic Benchmark also.
TA
The Admin
01:27:44
Agent Benchmark initial C initial 502. We said, right, we only check the initial 500 tokens. To initial 502. If the trace comes in, that will just destroy it.
SB
Sachin Bharadwaj
01:27:50
Oh, thanks again!
TA
The Admin
01:27:54
Okay, Amnesh.
AM
Avnish Midha
01:27:56
Uh, Ron. It would help, if maybe not now, but later, if you could explain what is distilling and how does it work.
TA
The Admin
01:28:04
Distilling is the data set that we're talking about. We created using Claude. Set it to allow, get the data, set it to hire, get a data, set it to medium, get the data that's listing.
AM
Avnish Midha
01:28:16
Oh, that way, okay? Asking the questions then? Yeah.
TA
The Admin
01:28:19
You know, asking a question in different formats and save it opens didn't have that kind of data set, right? But we have, you can. I can give you an assignment that ask 100 questions for each one of the email data.
🎯
Decision
01:28:29
That is, that is this.
AM
Avnish Midha
01:28:29
Okay. Okay.
TA
The Admin
01:28:32
Swati.
SB
Swati Bansal
01:28:34
Yeah, so, uh, when you're talking about Opus, isn't it like? Viet grain at the time of training. We are doing this comparison, but can this same comparison be also done when we are doing data cleaning? Because, ultimately, all we are looking for, is? High quality data, right? Or, am I mixing it up?
TA
The Admin
01:28:52
No. Selecting the best data while the running is happening. Like, what is the good quality data or when the model being clean is the question? Let's say we have a very good book for a one-year-old or two-year-old. Extremely well designed book taking that book and giving it to PhD student is not a great idea. So quality also matters. At the time of the training. Initially, it will be okay. It may be really good,
SB
Swati Bansal
01:29:15
Okay.
TA
The Admin
01:29:17
but not maybe okay.
SB
Swati Bansal
01:29:18
Understood.
TA
The Admin
01:29:19
That's the question about this answering.
SB
Swati Bansal
01:29:20
Okay, okay, got it. Yeah.
TA
The Admin
01:29:23
You? You answer this question why roads are bad in Bangalore? Moral training,
PD
Pranabesh Dash
01:29:26
And these companies, they distill the data and then trained. So, in case of India, if some company has the money, they just need to get the data. They can apply the same strategy. Chinese are better at hiding because anthropy will keep blocking that, so if. Okay.
TA
The Admin
01:29:49
isolation, and all of that is so far ahead. We don't want to fix the basics.
PD
Pranabesh Dash
01:29:49
Okay, so the willingness is not there and.
TA
The Admin
01:29:49
Your intent is not there?
PD
Pranabesh Dash
01:29:53
Okay.
TA
The Admin
01:29:55
And, and I blame it to our neighbors. What is our neighbor who are our neighbor of Bangladesh man, we have? No, we are not like they can't do anything. How many of us can they kill? There is no economic threat and in between India and China. God has artistic Himalaya. They can't attack properly. There's no threat to survival, like living in India, is peaceful. We have a tree. You have a food on tree. It falls on your head. You eat it and can survive. A question. Think about think of poor cold countries you can't.
01:30:32
You will come with examples that are wart on and all. That's fine, but there is no poor cold country. You can't be poor in cold country. Thing, so I would create infrastructure to survive. It is.
PD
Pranabesh Dash
01:30:39
You have to fight out here.
AJ
AJ Jain
01:30:49
If you give a sample to the model copy and we want to adjust the weights which are which are not getting which have the maximum loss, right? That's, that's the Opus thing. Now, in the always on to, I mean, in a previous session, you
TA
The Admin
01:30:58
Yes.
AJ
AJ Jain
01:31:00
mentioned ending data. We keep always on the give the data to it. So, the logic is.
TA
The Admin
01:31:07
Yeah, good.
AJ
AJ Jain
01:31:09
Sorry, the the logic is that? Uh, it was disregarding. Opus was disregarding the indict data. If you don't do always on because those weights were not getting impacted, the ones that had to
TA
The Admin
01:31:17
Correct?
AJ
AJ Jain
01:31:19
be. Reduced here.
TA
The Admin
01:31:21
See what what you need to understand is when we do the Opus thing, right? We need to give some samples to Opus. For it to calculate those weights.
AJ
AJ Jain
01:31:31
Yeah.
TA
The Admin
01:31:31
If you take 400 benchmarks? And samples from each 400 Benchmark, including indic also. Each Opus test is going to be so expensive that our whole training run is going to be slow.
AJ
AJ Jain
01:31:43
Okay.
TA
The Admin
01:31:44
So, amount of what is the amount of time you want to spend on tests is also important. Because Opus is expensive when you run this, let's say, for each each patch, we
💡
Callout
01:31:52
spend 35 seconds. Uppers will take 35 seconds, but I need to load the model I need to load a new data set. I need to calculate the results and to calculate the loss I need to calculate then 1 0 to 4 samples and then right. So, this how much you want to stop in between is a question, and that will be a very, very big question when you train because we want maximum throughput.
01:32:12
So, what we do only pick the best of the benchmarks? Which allow us to be fast, but that means that those best benchmarks are index and all will not even populate there. That's why we have always on philosophy, which makes sure that okay and indic and agentic is always on. We don't even care. We only focus on quality of data set. The map quality is there the? Good latex qualities there, and we focus on quality site compared to. Uh. Index file.
AJ
AJ Jain
01:32:40
Thanks, make sense.
TA
The Admin
01:32:41
Okay, Suresh.
SM
Suresh Mantha
01:32:43
Maybe a little of tangent topics, but how do we orchestrate security and ethics where will be trained and second?
TA
The Admin
01:32:50
That is the last one. Yeah, we have a specific stage for that.
SM
Suresh Mantha
01:32:56
F*** it?
TA
The Admin
01:32:57
Preference alignment.
SM
Suresh Mantha
01:32:59
And the second question is? Trying to include any kill switch that? You know?
TA
The Admin
01:33:06
Harness. There's a harness question.
SM
Suresh Mantha
01:33:09
Okay. So we can? Switch, which will typically stop the model or something like, stop responding.
TA
The Admin
01:33:18
That, that's a hardness question.
SM
Suresh Mantha
01:33:20
Okay, thank you.
TA
The Admin
01:33:21
Okay.
RU
Rahul Uniyal
01:33:28
Have some kind of a potential bias towards easy wins given across the stages.
TA
The Admin
01:33:33
All into the question.
RU
Rahul Uniyal
01:33:34
Suppose. Of, we are selecting based on the data. We are providing, but the model will also always choose whenever it is easy to update the gradients, right? And that might be the easy solution, uh, easy question or easy data set for data set, part of it.
TA
The Admin
01:33:57
The if the grade is a small loss is going to be small, we push for the high loss. We see what effects loss mode, not less. They can't select law. It's a math basically.
RU
Rahul Uniyal
01:34:09
But if I give it a hard problem, it won't be able to understand that and loss will be less, right?
TA
The Admin
01:34:15
Green will be high because loss is high. Heart problem. High loss, high loss, High gradient. Don't worry when we when we talk back application or not, you'll understand more. Okay, last question, chirag.
CT
chirag Tagadiya
01:34:39
I want to understand, for example, we have different different data, right, indic and agentic, and web and code. So when you start training, so, how do you introduce like which part of the data set at what level? For example, if you train for specific.
TA
The Admin
01:34:53
Back to your appointment, so you'll have to start thinking on that. I've answered some of the questions that there needs to be a stage, what stage it should be, something that I wanted to think on. Hey Raju. Come on!
RH
Raj H
01:35:11
So in all this? Is it me?
TA
The Admin
01:35:17
Good good?
RH
Raj H
01:35:17
Okay, so in all these courses, we haven't mentioned about any of the promptings that we must use, so those are not required like. Uh, chain prompting or self-consistency.
TA
The Admin
01:35:31
It is 20425, not, not 11 anymore.
RH
Raj H
01:35:35
Yeah, that is all right, but sometimes I find like CSS layout. If you mention something like that, it does do better work or something ever.
TA
The Admin
01:35:43
Best contact, let's move on the contact side. More on the rules side.
RH
Raj H
01:35:48
All right, great. Thanks.
TA
The Admin
01:35:49
There's hardness problem again, not a model problem. All right. Now, let's talk about curriculum, the order the model model learns it. And again, something similar that you're asking now. Modern pretending runs through deliberate stages. These are stages that we need to design. Training begins with a broad general text of established language. Let the model learn English bit of math. Let the model actually be able to speak, then only we can train it. Some faction knowledge. We have five fingers basic structure headers on top. Once the foundation exists, the mixture now shifts towards code, math, science, and reasoning. Heavy data set, so there's a gradual change, right, easier problem than harder problem. So, in MML Benchmark, also initially MLU, the MLE Pro, then Pro Max and pro Max 15, that Pro Max fold and Pro Mac Ultra fold, and so on. So on, so we need to structure our Opus data also like that. We didn't do that in week four. You're taking the whole Opus, put it together. Every quantum physics question is also there, and then we have the a string theory, a software format equation kind of question. Also there. At the same time, the long sequences are introduced later on.
01:36:58
And if only if you know the model can actually think you should ask a question that requires a longer structure, right? So, these stages are some of them are already there in your head. Also, if you think about it deep. Then, you can also come up with the structure, so each stage the data has a difficulty ladder that we need to find out somehow and starting with the simple material progressively generally going towards the harder and harder problem and longer and longer problem. So hard, problem. Long problem, as the model gets trained, we need to control that.
01:37:27
Right. So the difficulty band that we need to build and we need to stage why we are doing the pre-training part. Think of it, like this Nursery, then grade school, then High School undergraduate and graduate research, PhD. It's an easier way to look at it, but what does it mean something that you need to think on? Now initial seat. We are going to see we have General web. This is a example structure. This is not literally how we want. It's not that initially when the model is training in the b0 stage or the nursery example, you're just showing a for Apple B for more understand that way. It's the easy stuff that generally can be sent on right and the general stuff you're going to see that our stages are slowly, slowly changing.
01:38:05

[output truncated at 50000 of 96997 characters. Pass a larger max_chars (default 50000) to see more, or use read_page with a ref_id to focus on a smaller section.]01:38:35
So, once we have this ready, then you know that. Okay, now we can go ahead and train the model. Any question on this? Don't ask me, how do we build what percentage and all you need to think I need to think we need to arrive with the practical approach and hope it works because we don't have 10 examples or 10 steps that. Okay, let's try 10 different approaches and go ahead with it. We have only one shot.
01:38:57
Any question on this? No question good. Okay, now. So, now you understand the stages. Are there, right? So, we have stage one, then stage two, then stage three in the stage four and stage five. Uh, see General reasoning. Long context any any less? The best quality data for every example and hard but cleaned. And all of that, right, really, really the best of the best examples that we have is for the any last one percent of the training? While we do that, this also comes into picture. How do we keep the runs stable as we are moving along? So, what happens is?
01:40:00
Yes. Now, what happens is? If you suddenly change the data if you suddenly change the band, we have a band seed from the seed. We go to General from General we go to reasoning. If suddenly you shift, you're going to see our loss behaves like this. These are gradient you're seeing. So, then the gradients jump a lot. Right, and we can move all all of that. I figured out.
💬
Shias Abdul
01:40:22
Hi, I'm transcribing this call with my Tactiq AI Extension. https://tactiq.io/r/transcribing
TA
The Admin
01:40:25
And doesn't matter how hard you are. The top side will lose. We have 130 here. Mixture has changed only 18 percent. The sharpness is only 18 percent, and if you apply the shift, you're going to see that it's not 30X. It's still high. How do you introduce it? What the question I'm trying to ask is like? Uh, kid has crossed 12th class and suddenly you're giving a PhD level books.
01:40:48
Right. That doesn't happen if you see the if you think the education, how the education is also structured. Ninth is in the difficulty actually starts nine tenth nights, clear your math, science, or Commerce person, right? So, then, slowly, we have the 11th and 12th, 11th is harder actually compared to 12th, and then you suddenly you realize that what have you learned from the 11 classes? What the, what is required in the college? So? You need to have sort of a warm-up band.
01:41:13
Warm up is how much of the overlap is there? This overlap is required to make sure that suddenly you're not shocked. Right. It's a good example would be that you are right now in India and suddenly you're sent to Russia. And for the first few days, you are struggling. What do I speak? How do I speak? But if Russia Russian was introduced earlier in your language in your school, then you have some idea of how you can actually.
01:41:36
Speaker, right? So, if there's a gradual transition that has to happen, you clear on this. You can't just change the bucket. You have to have a gradual transition. You cannot introduce Sanskrit suddenly and say that, okay, start speaking shloka. It has to be gradually increase, and that gradual overlap is also required. So not only you need the band, you need to have a band overlap as well, so it is not a sharp line here. There's going to be some sort of diffusion of the band B1 and B2 going in also together, and this is a real experience by the way. I.
01:42:05
I don't have the graph. I should have added that, but in the V4, we saw moment which, in the band, woof, the loss goes up and down, and then suddenly we have to monitor. So, in fact? Good point. Should I open it?
💬
Suresh Mantha
01:42:21
is that instability threshold at 3x based on intuition/realworld observation.
TA
The Admin
01:42:29
Okay.
01:42:57
You see these ups and downs? This is actually what is happening still here. Not here. Some of the mid stages that you see is up and down. That is exactly where we are struggling on, handling that, and a lot of times let me find ex. These ups and downs that you're saying some some of the times. Here we are seeing that. Uh, gradients are jumping. So, we need to control it using LR. We need to control this. All of this will come in, is about to come in in future sessions. Some of this is coming because the LR some of this is coming. Uh, we can control using LR. We can control it using. This is jump.
01:44:08
I hope there. You see, that kind of job happens when we are changing something in the model. This is what I wanted to show, but I don't know full photo of it. Here's, I'll find more and share with you. But the point is that the fluctuation that is happening the gap of the fluctuation that is essentially the result of something changing in the model that it doesn't like, so we need to be very, very careful with the model that and multiple runs happening at the same time, right? You can see the average here in both the runs looks sort of similar, but the Gap that you're seeing how much shuffling is happening is the gradient we want to maintain that as well.
01:44:56
Sometimes the same trajectory is taking us somewhere else, and it will deviate later on sometimes debate earlier on. All of this has to be controlled while we're treating the model. Right here, using the loss. I hope that they could have read anymore. That is what I wanted to show you. You see this? This is the gradients that are going back. On average, we need to be somewhere around 0.2 if you can maintain 0.2 gradient. You know that you can sleep properly initially. You can start learning is going to start very high, but later on, you just get stable. I am just slowly going up, which means model is struggling, and you can see that here it stabilizer 0.35. Not a great idea.
01:45:28
Right? So all of this is something that we have to discuss. You have to get slowly back into. Right. So, what I was basically said is that? We can't have this certain shift in the. But we have to have a very smooth band otherwise. Moment you have something like this. There's a very high chance the model will deviate certainly shocked. Oh s***. In college. Youtube.com. He started Bengali. That's where my love for my Bengali friends started. Yeah, you have a question.
01:46:32
Yes, sir, can you please speak in English? Imagine speaking in class of 400 people. Requesting the professor. Oh sorry, sorry. Then, you'll start a question. Then, look at the Bengali, something I can't. I don't even remember it. What do you think in Bengal? And then, they'll start discussing Bengali. My pain is your pain now? So here is your long assignment.
🎯
Decision
01:46:57
So, this is where your thing your thinking starts now. I hope you realize this is the most consequential artifact that we we have to design. Everything else model I'm telling you. I've talked it multiple times, and you will also get it immediately, because now I have so much background as well. This is the most important decisions of our life for the next four or five months. This is going to decide our model is a s***** model? It can actually do something in life, or it is going to be just one more model like just exists.
💡
Callout
01:47:25
Each one of you is going to draft a mixture of curriculum plan for the va5 as a return specification. I want the plan from you. You can read the whole damn thing. This is, I know you're going to copy, so you're Claude anyways, but the idea is that I want full, end-to-end, verifiable, translated, synthetic layer tier from you. What do you think your plan is going to be if you were to do this? Now you know benchmarks on, you know everything? How we'll come up with this. When are you going to train on a long context? Where are you going to train on agentic when I'm going to train on reasoning, where am I going to train on Sanskrit if ever, or Urdu or other languages? Right, so you need to come up with everything.
01:48:02
Now, this is important because this is how the evaluation will happen is a subjective evaluation. So don't say that I got 200 that produce. Don't complain
💡
Callout
01:48:11
that. You have valid and how well your plan will hold up. If a reviewer not seeing someone from open ASR across and of course, they will be smiling if you sit around them here. Data and data curriculum. Review side across from you and pushed on every number. So, the great rest on the quality of your reasoning and the evidence behind the choice. Why do you think what you have selected to train on comes last and tightly argued? Short plan is going to score along the.
01:48:39
Uh, padded earnings or nothing basically earns more points or not. A strong submission gives a defended share of budget to every capability Lane and States. The index rate across is verified, unverified, translated syntheticia. Whatever you can think of, I wanted to share this because this is what I will be using to. Evaluate your assignment. Your assignment really is to come back with this.
✅
Action Item
🎯
Decision
01:48:59
What is inside all of these? Got it. And believe me, this you are looking at the Opus 5.5. 4 + 1.5. Maybe charging 3.5 plus 1.5. I might also play better. Or open for any question you have. What I said meaning, by the way, a simple GitHub GitHub readme. It may have your
✅
Action Item
01:49:33
code for calculating something, something to some sort of script that you may have run and think, but I'm looking for a readme link, this time, not a netlify app or widget. That link is. Our definition of how you decided that that's the code you want to do, and that's the. Pipeline you want to run on, or you want to train on, right? This is the. And one thing that I've not mentioned. But probably, I should think, also long context means, what? That each sample is going to be longer, right? And this is what we did earlier in our. V4.
01:50:08
So maybe of interest you? Uh, I don't remember exactly, so I may be wrong, but it gives you a general principle. So, in our case, we had a one billion, which turned out to be 2 billion. Then we had our three billion dollar five billion. Then we had our five billion, which turned out to be 8 billion, and uh 70 billion, was the initial plan, which turned out to be forced to be 120 billion. So, forget all of this.
01:50:28
We had a Model A, model B, model C, Model D. Okay, we train, I think till this stage at 4K, because it was fast. 4,000 words a lot already. Try writing 4000. I don't know how, how many of you would have written four thousand words? Thinking yourself? Actually, that's how we increase. So, if we were to move longer, then you will have 16 con. What is it this? K the context line 4000 token examples 512 examples of 44096 tokens, then finite 12 examples of 8 000 sequences right, and then 512 examples. Each batch is 16, 000 is going outside the screen and came back from here. You saw that I can draw around the screen see. No to the 32k.
01:51:18
Then 64k? Can you see that I can move in 3D 4D? So it stopped here, right? So When you say long context, that means that you are doing this. You are in this
💬
Adinath Auti
01:51:32
notion ink magic
TA
The Admin
01:51:32
so? Here. When you are discussing long context, you need to think about that that. Or token. Length of each example is increasing. That's the meaning of also a meaning that you need to keep in mind, right? It's not that a bash. The sequence lens is 4K. You have 100K example. You're going to cut it and train it. You've lost the whole memory, so you have to train when you say 100K context. You have to train 100K, otherwise you can't prove it.
01:51:59
You can prove it, but you have to turn on longer, maybe half, then proven full, all right open for questions, if any. No questions.
RH
Raj H
01:52:20
So, last time we'll use like 20 million or 40 million tokens. This time, we don't say how many minutes.
TA
The Admin
01:52:29
I didn't get the question.
RH
Raj H
01:52:32
The previous assignment we took like 20 million or 40 million to train our data.
🎯
Decision
01:52:40
The stem how much you are not specifying, right?
TA
The Admin
01:52:47
I think between 2.4 to 4 trillion tokens. We'll extend it, but you don't have data set for extension also.
SB
Sachin Bharadwaj
01:53:03
Around for the long context part of the curriculum. If we train it like, you know, if you have 2K, 4K, 8K, 16k in order of? Uh, and keep up a ratio of the particular long context. Data set, uh, in place, right? Is that a good strategy for, uh, for generalization across and up till 1
TA
The Admin
01:53:16
Are possible.
SB
Sachin Bharadwaj
01:53:19
million context?
TA
The Admin
01:53:20
Not possible because you cannot have a batch that has a 4K 8K16 example in a batch. All example have same length.
SB
Sachin Bharadwaj
01:53:27
Yeah, so I'm saying I will clap to the samples of 2K only, and then I have a separate batch, which 4K only and stuff like that, right? So it's like a.
TA
The Admin
01:53:37
Yes, come on.
SB
Sachin Bharadwaj
01:53:38
Sorry. So, is it a better strategy or you just? Okay, it will generalize right up until the maximum context line.
KS
kamran shaik
01:53:53
Started, right?
TA
The Admin
01:53:55
There is no shorter one. Shorter one is a loss of compute for us, so we don't
KS
kamran shaik
01:53:56
Okay, okay.
TA
The Admin
01:53:59
make a shorter one 5, 6, 10, okay.
KS
kamran shaik
01:54:04
So, we deliberate it to have at least 4096. Okay.
TA
The Admin
01:54:09
It's very expensive when you run, and then you, you see that it will hurt a thousand dollars.
KS
kamran shaik
01:54:17
Hey, one more question. Yeah, so. These kind of? Uh, messages right, wherein you have system message role-based messages in the context. So, those are part of sft, right? Not, not free training.
TA
The Admin
01:54:38
I didn't get the question.
KS
kamran shaik
01:54:40
Uh, so, where wherein? We have these, uh, role-based messages, right, a system user? Uh, tool, assistant, etc, etc, the two intro conversation. So, that kind of
TA
The Admin
01:54:53
Correct?
KS
kamran shaik
01:54:54
conversation is happening in sft, part of free training, okay?
TA
The Admin
01:54:57
Yeah, so today we discuss Sage two stage stage 4. From a needle to here. Swati.
SB
Swati Bansal
01:55:07
Yeah, I mean, not related, but, uh, the issue that happened with open ai's model breaching, you know, hugging face, uh, from that, uh, hacking issue was that the
✅
Action Item
01:55:16
data problem, or was that a hardness problem?
TA
The Admin
01:55:20
Workers told me.
SB
Swati Bansal
01:55:21
Because.
TA
The Admin
01:55:22
Honest problem. Honest problem. See when. So, the objective is, uh, you tell the LRM that given a problem. You have to solve it. That is, how people solve. Let's have people train it the given a problem you have to solve. We never add that intuition that if the problem is bordering around. Nuclear fusion, or how to make a nuclear bomb don't work on it because they want it, at least the yield government to train on it, right? You're going to use it or some people to use it, then becomes a problem, how access stop it?
SB
Swati Bansal
01:55:51
No, but. But if you had training examples around reasoning that you know, you got to stop. You know if, like, based on?
TA
The Admin
01:56:01
No, that's not you, can't? You can't tell that tomorrow a model does not have a
🎯
Decision
SB
Swati Bansal
01:56:02
Um, yeah.
TA
The Admin
01:56:05
rule book. It has a full functioning brain. She can't. I, I can't say for all the problems. Let's say we say that, okay, nuclear don't work. What about biological, say? Okay, biological don't work okay. What about inciting people of towards violence? Okay, don't do that. Also, okay, what about, uh, asking kids to play blue whale. Okay, don't do it also, so the list is just not possible.
01:56:28
Can I always come back with a way of getting it to do something wrong? That's why it becomes a hardness problem. The hardness understands something wrong is happening. Let me solve the model.
SB
Swati Bansal
01:56:37
But it was the reasoning that could not be controlled from by the harness, right?
TA
The Admin
01:56:42
Reason will not be controlled by the harass will see model doing something bad and stop it. That's right, there is no way that.
SB
Swati Bansal
01:56:47
Oh God.
TA
The Admin
01:56:49
Uh, for example, every human has done a mistake. Even though we are the most sophisticated? Empathy, embedic, human empathetic animals.
SB
Swati Bansal
01:57:00
Right, right? Got it?
TA
The Admin
01:57:01
Nothing should not be there in the model, but I'm saying the heart, it's a hardness problem. The harness has to stop it.
SB
Swati Bansal
01:57:08
Thank you.
TA
The Admin
01:57:09
Not at all.
SB
Sachin Bharadwaj
01:57:14
The sample training sample? Uh, and if you have a large, you know? A lot of samples to be evaluated sparsely. If you have a large model also to begin with, but let's say we distill the model. I have a smaller one. Is it a good proxy, uh? For the online model that we are evaluating right now, the office? Weights are completely changed.
TA
The Admin
01:57:34
Creates a completely changed, a very different capacity check.
SB
Sachin Bharadwaj
01:57:38
Okay.
TA
The Admin
01:57:52
Correct?
KS
kamran shaik
01:57:57
Understanding on the Opus, so it's like we have a map of all the weights. Uh, in that particular model. And then we run an algorithmic. Check whether the back propagation is affecting the same proportionate of the magnitude of weights. That. Is affecting in the, uh? Benchmark data set, right?
TA
The Admin
01:58:22
Right. So basically, like, let me again Reiterate. The logic is. I have a test. I see my model is good or bad in that test. That good and bad is a number which is loss, but inside the model, I find out the important ways that are still bad for that particular Benchmark.
💡
Callout
01:58:39
Now, when I'm running a test on 124 samples, I see which of those samples are also bad. That is a hint that those examples that are bad for my test. Are good for training.
KS
kamran shaik
01:58:53
Got it.
TA
The Admin
01:58:54
And I said bad, like, I meant higher loss here, right?
KS
kamran shaik
01:58:57
I had lost. Yes, yes. So, so one more thing you spoke? That it is for very expensive for our training cycle, but this operation has to be done only once, right? This is like.
TA
The Admin
01:59:10
No. But now it is bad in science. Are golden boxes also changing, right? What Benchmark it should be good at differences is changing. So, as Opus is changing, it needs to check the model's current state. Is my model currently bad in math? Is my model currently bad in science? That is, how that is, why it's expensive has to run regularly, if not once? What's with me? I have the weight weights of the full life cycle. I don't have
KS
kamran shaik
01:59:41
But.
TA
The Admin
01:59:44
that I only have the current weights. I don't know the future weights.
KS
kamran shaik
01:59:48
But the data set says that we had originally was fixed at the beginning, right?
TA
The Admin
01:59:56
Yeah, so that's why we have to keep the fraction. So let's say we keep the 50
KS
kamran shaik
01:59:58
So, we are going on filtering some portion of the data set at each and every iteration, so it makes the data set smaller.
TA
The Admin
02:00:10
fraction. So now that this means that if you collect one tail end tokens, actually, only half a half a trillion token will be turned on.
KS
kamran shaik
02:00:20
Okay, but what I'm saying is we are filtering on the same data set again and again, right?
TA
The Admin
02:00:25
Not again and again. You never pick the same batch again, right? You throw that part.
KS
kamran shaik
02:00:29
Oh, got it.
TA
The Admin
02:00:31
And this is the actual Opus paper. The their claim is that they get 8x efficiency, which means the loss we are going to get at 20 billion tokens would have taken 160. That's why this is amazing, and I could see because the total number tokens we collected around a trillion tokens actual training was 200. And of 200 somewhere on I think 40 or 50 were opens 40 or 50. Let's say 50 of Opus 15 to 88 it's 400 billion, so overall training run that we did was around half a trillion.
02:01:00
Right, and if to get to the same loss, otherwise, we would have had to train? Uh, how many? 8 into 15 half a trillion tokens? All right guys. I'm going to stop the recording, uh, if you have any questions I'm still here. Right, so very, very important that you get this right. You design this properly. Or get some politician in India to understand this. So, they understand what actually needs to be done.
💡
Callout
02:01:44
Yeah, this is a simple plan here. This is General reasoning longer text any
💡
Callout
02:01:48
link. 550 ministers in Parliament, solving this for a year. We will make what Opus will make Titan 70, right? So just come together, solve it, how difficult it is. We have to live in Destination also.
💬
Siva Ajjapu
02:02:29
Please publish the session material.
TA
The Admin
02:02:34
Studio, please publish. Yes, you're right, I should publish. Uh, that.
US
Udit Singhania
02:02:51
In this assignment. So, basically, the myths you showed on the high school and
🎯
Decision
02:02:57
the gray grade and the research level and then Middle School level. So, for each
💬
Suresh Mantha
02:02:57
sab paisa aur power hai saab... ye sab neta ko kyoun chahiye
US
Udit Singhania
02:03:01
level, we have to explain on all the con different different, uh, tasks we are performing. So, for the how we are using the percentages, and what are the percentages in its verifiable? That's that's what you have to do it, right?
TA
The Admin
02:03:16
That's correct.
US
Udit Singhania
02:03:19
And also the long, long context, one the separate one there. Okay.
TA
The Admin
02:03:19
Yes, why? You can just, you just can't say 50, that 40 percent that I need?
US
Udit Singhania
02:03:29
Yeah, when detail in the Bible, like, why, how, how we next, we reach it and why
TA
The Admin
02:03:30
Correct?
US
Udit Singhania
02:03:34
we reach that? Okay.
TA
The Admin
02:03:36
And this, uh, this, uh? Assignment essentially see last three, four assignments. Are you guys actually coming together and giving? Me ideas. So, I'm taking best of that and building a strategy.
🎯
Decision
02:03:49
Right. So, if your studies are really good, that becomes a part of training. So, I'm exporting the job of essentially deciding what to do so, we all can actually contribute. And this, this is the best way I could find because otherwise, I will write the code I'll share back with you, and then you can just say that. What did I do? But because of the depth of number students, it's possible that when the solution comes back to you, I may not be able to say that this solution came from AJ. So apologies for that in advance. Uh, AJ.
AJ
AJ Jain
02:04:22
Yeah, so Rohan, uh, what you were explaining before this one Opus? Essentially, it's a it's. It's a matter of the quality of token, and actually, that is also. Find it. What I'm trying to say is. We know the different areas that we have to train on coding, reasoning, General web language, those sort of things. Depending on the. Embedding in the tokenizer that we learned? Like the in the previous run, you cut down the 1.2 trillion to 200 billion for V4.
02:04:55
So, so there is a nite thing. I mean, when we say we have to go and figure out the data? Uh, you know, new data, which which we can train on, but that is finite as? Uh, when training the model, I don't know if I'm articulating, and if you're getting my question. But if it's a matter of quality of token that we already have probably for coding and reasoning? Then getting. We want to get more, you know, different data, variance of data, or within that same thing. We need to figure out quality tokens.
TA
The Admin
02:05:25
Okay, so let me answer this way. First of all, we have done downloaded data. Okay, we know Opus is going to be there. Let us say. We know Opus is going to throw a 50% data away. And now we want to train a model on 2.4 trillion tokens, which means that we need to correct. Open it. First Step. Second, we have cleaned it. We duplicated, and we made our data quality good.
02:05:48
But do you think at the latest stages when our model can actually write the whole GPU kernel, telling it how to write a Pythagoras Theorem. Python Loop is a good idea.
AJ
AJ Jain
02:05:59
So that I understand everyone. That's, that's my question. Actually, sorry.
TA
The Admin
02:06:03
Because when we are making these batches right, there is no way we can do this free hand. One of the mistakes I did in V4 was we, I forgot to save the Opus selection. Had I saved the open selection, I would not pick the other example it threw away. So, this time, we get to do that, and we run this three four times on our five trillion fixed data set. We know these are the good samples.
02:06:30
And because open air and charge option, others have done so many times. They are throwing away samples and all they know I don't need to train on it. So, the use of Opus also drops.
💬
Mukund singh
02:06:38
The govt. had shutdown internet since last afternoon at my place. Can I submit the session 4 assignment by midday today?
✅
Action Item
🎯
Decision
TA
The Admin
02:06:39
I hope what you get, what I'm saying, like, it's literally the both example.
AJ
AJ Jain
02:06:42
I do.
TA
The Admin
02:06:42
Yeah, when you open a book, for example, do you read the preface. Do you read the publisher note and all not required? Throw that away. Make. Keep the contracts on the main thing.
AJ
AJ Jain
02:06:54
I think. My question was, let's say, reasoning for that. We give some logical problems to the model to train on to build that reasoning, uh, muscle that it has.
TA
The Admin
02:07:02
Yeah. Yeah.
AJ
AJ Jain
02:07:05
So, we have one trillion tokens on that, based on the first 200 to 200 billion tokens. It has figured out reasoning. So now for the balance 800 billion and suppose we have. We have gone to 500 million which Opus has rejected 200 billion is what it is trained on, so the balance 500 billion it will it improve that reasoning muscle if we keep giving it that? Or do we have to do that is?
TA
The Admin
02:07:31
No, no, no, so okay?
💬
chirag Tagadiya
02:07:37
shameful from government, which state ?
TA
The Admin
02:07:45
No, I got a question. Let's say we are at this stage. Okay, let's say we are at this stage. Okay, now General Web 18%. This is how a baps looks like genovap is 18, 30 is code, and 16 percent is reasoning 18 percent is long. Context 12 percent is indic, and then we have stem. Let us say indic is part of OS also okay. Now, what is Opus? Opus is a test on 12345 and 6 benchmarks.
02:08:17
And you're saying that long reasoning we have excelled already clear?
AJ
AJ Jain
02:08:24
Yeah.
TA
The Admin
02:08:24
That means our loss is going to be higher here. Which means off the 100 samples. It is definitely going to throw away the long reasoning, which is this part.
AJ
AJ Jain
02:08:36
Okay.
TA
The Admin
02:08:36
Which it is going to pick more samples from where our model is bad.
💬
Mukund singh
02:08:37
Bihar
TA
The Admin
02:08:40
Again, it automatically solves the solves. The problem we're talking about?
AJ
AJ Jain
02:08:44
Okay, good.
TA
The Admin
02:08:44
Right. It's a collection of defensive if I have Excel python already, and my loss in Python is not not reducing because of the sample. I know I've already accepted, and we're not going to take that. I'm only going to pick the fifth top 50 five percent 49 that we need to identify of the samples that are affecting the benchmarks. So that it's a live test. I'm going to spend more time learning or reading or calculating laws for the examples, which are still helping me improve my benchmark score.
AJ
AJ Jain
02:09:12
Sorry! Thanks!
TA
The Admin
02:09:13
Okay, somewhere.
SK
Soma Korada
02:09:20
I am still trying to understand this Rowan. Isn't it like a chicken and egg problem because we will know, uh, that only when we are training, right, like, um? Where, which data sets or which benchmarks? Our model is not performing good. And we know that Opus will remove data accordingly. But then, if we have already set these percentages up front, like our data set is going to have in this proportion, the? Uh, data.
TA
The Admin
02:09:53
We don't use it randomly.
SK
Soma Korada
02:09:53
Correct the even though we say every stage has it. But yeah.
TA
The Admin
02:09:55
We are not saying every batch has this. We are saying every stage has this right. Answer the answer this answer. And only 24 hour indic. I had one, two, three, four, five, six different components in my back supposedly. But randomly, this is possible. Now we have Opus. Opus was given long context Benchmark indic Benchmark python Benchmark C, plus Benchmark agentic Benchmark, and so on, and so on. It will get data for long context also.
SK
Soma Korada
02:10:38
Yeah.
TA
The Admin
02:10:39
So, from this thousand, we select 500 of the best ones. Don't matter. It has the python or C doesn't have to have. If it had, it will figure out that it is required or not. So the Opus golden proxy, when we discuss Opus that golden proxy, is the key. What all do we add in golden proxy? The golden proxy should be a representative of something that I will find in my data set in that batch and filter out that thing in the batch.
SK
Soma Korada
02:11:07
So that, uh, when you say golden proxy, you're referring to the benchmarks.
TA
The Admin
02:11:12
The benchmark will train on and the size of and the selection of the samples in each of the Opus test.
SK
Soma Korada
02:11:19
Okay. Got it. So if our data set for a particular Benchmark, like the coding Benchmark, if our coding samples are very less than the data set, uh, eventually our model may not perform on that task at all. Okay. So based on what we want our model to do, we have to choose the these percentages accordingly or the data accordingly.
TA
The Admin
02:11:48
Correct?
SK
Soma Korada
02:11:51
Sure. Thanks!
TA
The Admin
02:11:57
All right, I'll see you next Saturday. Seriously, it is more subjective in nature, but it requires your eating. I want you to read through the first part and second part. How do you actually select what is your mind telling you? What are the examples and examples and ideas. You can come up with, and uh, if it is really great, we'll start pulling it in and then start making the world policy Suresh.
✅
Action Item
🎯
Decision
SM
Suresh Mantha
02:12:22
Is it possible for us to, you know, create our own what it takes for us to create our own indic language benchmarks? Do we need to take it a bit? Set list for one language. Can we give it a try? Okay. Have a great weekend!
SB
Sachin Bharadwaj
02:12:47
This one more question, uh, for Opus? Why is loss? Not a good proxy from the previous checkpoint, uh, why do we need to go do a backward pass and look at the cumulative gradients?
TA
The Admin
02:12:56
Because the model has changed.
SB
Sachin Bharadwaj
02:12:58
It's only because we are looking at, uh, some steps backward in history, right?
TA
The Admin
02:13:00
New model.
SB
Sachin Bharadwaj
02:13:02
That's.
TA
The Admin
02:13:02
Correct?
SB
Sachin Bharadwaj
02:13:03
But the so in the. In that case, the gradient is the latest. I mean, latest proxy, that some some sort of that right?
TA
The Admin
02:13:10
Correct. It's like you have a kid who was bad in math. You gave all the math tuition arts good in math, but you're looking at the ninth class grades and
SB
Sachin Bharadwaj
02:13:20
Yeah, yeah, yeah, thanks.
TA
The Admin
02:13:20
again putting them back in the Massachusetts. He will cry, right? Okay, yeah, sir.
YR
Yasir Reshi
02:13:27
Of prime importance. We've seen deep seek coming in with the architectural advancements or training pipeline advancements that has also optimized the whole thing. So, I mean, while we say that China has taken the data or that distilling thing, but there is some intelligence in the architectural and training pipeline aspects also, which, uh, kind of, makes them stand apart at times. I'm just giving a view.
TA
The Admin
02:13:52
I'm just giving you a view. I'm actually giving you a view.
YR
Yasir Reshi
02:13:58
Okay.
TA
The Admin
02:13:59
What you need to understand is the China is China and us. Both of them are on a different trajectory. The US is through. Money. Doesn't matter the architecture, how much it cost. I need the model to be really, really good. China says that I want to make sure that it costs really, really low. So more people can use it, and China seems to be winning because that's a winning strategy. You can see how close they came here, but now the Gap is increasing because the quality of data is changing someone's talking on the phone. You can talk to me also in the class. Right, so that has a gene, and that is not happening.
YR
Yasir Reshi
02:14:30
So, but from a commercial perspective. Oh, we are talking about open sourcing or open waiting things. Uh, China is maybe doing more. Service to the humanity for that aspect.
TA
The Admin
02:14:44
What is the meaning of Open Source weight? Open source weight means that here's the weight ticket, right? You can also run the same model now. Do you know the
✅
Action Item
YR
Yasir Reshi
02:14:49
Yeah.
TA
The Admin
02:14:52
amount of money it takes to serve a model serving a 2.4 trillion token model? You need a massive 100 billion dollar cluster. You and I come from it today. And China is not releasing small model that we can request. Of course, there you can take coin and install on your laptop and work, but how many people are actually using it? Open sourcing is a very different way of.
02:15:12
Uh, challenging what is happening, because that allows internally China to different companies in China to take those models and actually build on top generate. Their own, raising thoughts from that. A very different way of attacking the same problem.
YR
Yasir Reshi
02:15:27
That is surprising that in India, also, there are many companies, but we are not
💬
chirag Tagadiya
02:15:31
serving is also enginerring problem
YR
Yasir Reshi
02:15:31
able to do much in this space. Uh, model creation or architectural upgrades or advancements that the world will use then or replicate.
TA
The Admin
02:15:42
Yeah, and that's the reason. Let's look at.
💬
chirag Tagadiya
02:15:44
rokda
TA
The Admin
02:15:49
Act and look at the problems that we are discussing. This is my channel. Try and follow a lot of stuff. Bangalore construction legal problem, then someone is selling something. Bag manufacturer. He's looking for a technical co-founder. It's called orange Cube electron. It Cell exposed? By the way robotics part, but if I just focus on, let's say, what is happening in Bangalore? I stopped asking when I start.
02:16:23
Where is someone? Doing something really, really cool and?
YR
Yasir Reshi
02:16:40
One more thing on the ethics side, uh, ethics or conscience, that kind of thing. So, we are not attempting or making models more ethical or. Have a conscience that kind of training. At what stage we can see if I put a message on GPT whether the president was resigned on moral grounds on ethics, because there is something that has happened in the in the ministry that he is heading.
TA
The Admin
02:17:07
See.
YR
Yasir Reshi
02:17:07
And it should say that on on the basis of morality ethics. It demands that he should sideline from the ministry.
TA
The Admin
02:17:13
See. It's very, very controversial thing. I'm about to say, and, and I'm saying it, because I recently saw something, and that put the question back in my head. Now! If you see ncrt, I've gone through the whole ncrt and I was in the Hindi medium for history. At least still history was being taught, so I remember that even more. The way the Mughal adventure in India was shown was that the plunderers came into India and then stole money and went out right. They killed a lot of Indians and it. It literally mentions in books that a lot of Muslims and Hindus were killed at that time.
02:17:49
But if you go to the biographers or the guys who came in? They are very clear. They killed the coffers, and at that time, the cafes were Hindus and the Buddhists and Giants and other religions that were there they were killed. They were killed because. Uh, it was a what is called Jihad kind of thing that happened at that time, right? But the books were written in such a way that is sort of.
02:18:10
Diplomatic in nature. It doesn't hurt anyone's sentiment today, and then it goes forward. Now, the question goes back to you. Is that ethical? Should we hide the actual history? Should we show it? Should we tell it to the kids who are in school and learning about it? Should we take a risk today to explain to everyone and create a yes we should? We should forget.
💡
Callout
02:18:30
We should not let any others forget. We should not let Jesus to forget. You should remember what happened 1944, but we should forget because that happened in 200 years ago. There are these questions where the answers just don't exist. I don't know the answer. I don't know what is it, does religion, and it has a player also today. Again, a question that, I know, don't know what if I were to teach my model. I will make the model and I'll say until listen, but at least not love any religion in that case might come back and say no religious. B*******, then you're going to say the moral is saying, and the RSS will leave behind and.
02:19:00
It is difficult to ask. That is why this stage exists, which is called the preference alignment. For a country, we need to understand the preferences of the country and the country's professionals changes. Right when the India Pakistan war happened, then we were in love with the government and what they did the neat paper released and how the actions it took now. We are against that, so the preferences keep on changing. It's a dynamic thing. What you're going to answer your wife today is a different.
02:19:27
Answer based on how she looks how she's looking. Is she angry, she making fun of you? You get the answer. It's not a simple thing. That is why it's more than the preference, alignment, and the harness problem. I'm trying to make, for example, a simple algorithm which can help you identify the. Ah, radar cross section, Fable said. I can't help you. It's against the US military policy.
02:19:56
Now, I'm thinking that why kind of able to do it and I understand their point of view? Why would Fable let me design? Uh, anti-radar system. You get the point. Wave. The line is drawn, will depend always on the country, what the country wants, so we sitting here and talking about ethics of the AI model that us is making that the US. I think they will follow.
YR
Yasir Reshi
02:20:18
Okay, so preference alignment is where the guardrails can be. Yeah, okay.
TA
The Admin
02:20:23
What else are implemented? Shape, style, helpfulness, and safety.
YR
Yasir Reshi
02:20:26
Okay.
TA
The Admin
02:20:29
Rahul.
RU
Rahul Uniyal
02:20:31
Uh, one confusion I have when we are going away data and Opus. For a particular stage. Will it be thrown away for complete process, or it might
TA
The Admin
02:20:43
No, we throw it away. We don't look at it again.
RU
Rahul Uniyal
02:20:43
be used for next stage?
TA
The Admin
02:20:47
Because model model's current weight are saying that I've learned it already.
RU
Rahul Uniyal
02:20:54
Another question, so of reasoning. We might have a lot of data set for the reasoning for the English language. But not for the endocrine. So are we trying? Are we trying to? Get synthetic data. Try to generate this kind of a data for indicate language.
TA
The Admin
02:21:12
No, I mentioned that earlier also that. In a model, we have three parts. This is where the token to logic conversion happens. The logic thinking happens here and then logic to token conversion happens here and goes out. So, this is common for all languages. So, even if you're talking in English or Hindi or Malayalam or Telugu or Urdu or Chinese, the conversion has already happened into logic at this stage. And here is all logical thinking.
02:21:43
And after that, the conversion back happens here. So, if you're training on English long context, that will have a huge amount of back for Hindi automatically. Having said that, had you had Hindi long logic conversation, then it may add more, uh, local context in which you need to think about things, and then it can think better and give you a better answer so. Uh, 90 taken care to push really to. At English level, you need you do need 10 percent sympathy.
RU
Rahul Uniyal
02:22:08
If you're saying 9010, in that case, the reasoning capability of English took secondary language should be 90 to 10. That's all right for.
TA
The Admin
02:22:18
90, yes.
RU
Rahul Uniyal
02:22:21
For the current winter model.
TA
The Admin
02:22:23
Uh, adding it?
AA
Adinath Auti
02:22:29
Uh, so the um? Always on data set, right? The always on part of the data set that is not a filter using Opus, right?
TA
The Admin
02:22:37
Unfil. Index.
AA
Adinath Auti
02:22:40
And the indic data set will reside in that part. Yeah, as as we go through, um, like the different phases of our training, uh? The index part will go on increasing because we want to emphasize that in the later stages.
TA
The Admin
02:22:56
Uh, we will always keep it constant, so it's not out. We're not including, we are not going to increase it. We'll all it will always be there basically.
AA
Adinath Auti
02:23:03
All right, but, uh, but because since it's not being, uh, checked by Opus, won't there be a possibility that some of that data will be repeated?
TA
The Admin
02:23:14
That data will be repeated if the data is less. That data will be bad if you have not done a quality check.
AA
Adinath Auti
02:23:20
Okay, so if let's say we don't, uh?
TA
The Admin
02:23:23
So, we have a four four trillion token Target, and we say that eight percent of that needs to be index. Then, 320 billion tokens need to be there, cleaned.
AA
Adinath Auti
02:23:33
Okay. All right.
AJ
AJ Jain
02:23:43
General question just just thinking? Um, and as part of this course, we definitely want to build a model from scratch. Uh, you know, to learn, essentially. But uh, you have your V4 model, and a model is basically the weights that it was trained on and where it was calibrated. Uh, so is, is this an approach where you have the V4 weights? And then you using Opus. You supply the new tokens and data. Uh, and then figure out what all things we can do to improve that model. So that it performs better on different benchmarks.
TA
The Admin
02:24:17
The DNA is already locked in. You can't change the DNA.
AJ
AJ Jain
02:24:20
Oh, is it okay?
TA
The Admin
02:24:21
Yeah, once a model is trained strained, you can't untrain it on a few things and impossible. Will cost more actually.
SB
Sachin Bharadwaj
02:24:34
Corpus of training result. Let's say one trillion tokens right now. How do we decide the capacity of the model? Is it only a function of of dollars or there is. There is a lower, uh, lower floor. It has to be higher, at least from at least four billion, eight billion 12 billion. I mean, I know, the chinchilla paper stuff. Can you? We have more insights on that.
TA
The Admin
02:24:54
I didn't get the question.
SB
Sachin Bharadwaj
02:24:55
So, let's say, I have 1 trillion tokens to train a model. Now, what is size of, uh, what is the parameter size that I chose, I should. I choose 32 billion parameter model. Should I choose 120?
TA
The Admin
02:25:04
Budget. Question question was it? Ginchilla says 20 people are somewhere around 65, 40, 45. 40s is a good number to Target. So, if you're saying one trillion divided by 40, that's 25 billion parameter model that you should have. But if you have money to train that. Also, if you have less money, then you will increase this and reduce the size of the model. Then you can train 10 billion, so it's a function of money. How much money do you have?
SB
Sachin Bharadwaj
02:25:34
Okay, but Okay, and it has two at least greater than certain certain capacity size, right, at least a bit better than 8 billion for it to generalize. And all that stuff, right?
TA
The Admin
02:25:43
Correct?
SB
Sachin Bharadwaj
02:25:44
Okay, thanks.
TA
The Admin
02:25:46
Both kinds there, but most are free only.
AM
Avnish Midha
02:25:50
Rowan for the final like certification for that Benchmark. We ourselves do it, or is it like when we submit the chartsum fee and they do it?
✅
Action Item
02:26:01
Most are free. Okay.
TA
The Admin
02:26:03
Mustafree.
AM
Avnish Midha
02:26:03
Okay, good.
TA
The Admin
02:26:03
And it's a web service. Basically, it's not costing them anything.
AM
Avnish Midha
02:26:07
Okay, okay, got it.
TA
The Admin
02:26:12
All right, I'll see you next Saturday. Have fun, uh, focus on this assignment
✅
Action Item
02:26:17
and try and think. While I know your agents are going to be working on it, but?
🎯
Decision
02:26:23
Mukund. Most of the time favor will fail. Answering some of the questions that we are about to ask, so make sure that you do read what it is writing. It might
💬
Adinath Auti
02:26:29
enjoy your b'day chief
TA
The Admin
02:26:31
sometimes simplify it and. Yeah, this is it. If we get this right then? We can do something good.
💬
Yasir Reshi
02:26:39
Have a good day
TA
The Admin
02:26:42
Have a very beautiful day. I'll see you next Saturday. Yeah, I'm about to be born.
SK
Soma Korada
02:26:47
Enjoy your birthday, Rohan.
💬
Shwetha D
02:26:50
Thank you Rohan :)
YR
Yasir Reshi
02:26:52
A.
TA
The Admin
02:26:54
Good job, take care.
AJ
AJ Jain
02:26:54
Thanks!
SK
Soma Korada
02:26:55
Q.
TA
The Admin
02:26:55
Thank you. Thank you bye!
💬
Kunal Sinha
02:27:02
Happy Birthday Rohan :)
