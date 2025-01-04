from agora.auxiliary.episode import Episode,EpisodeTest
from .broker import Broker

Episode.broker_cls = Broker
EpisodeTest.broker_cls = Broker
